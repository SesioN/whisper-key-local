#![allow(dead_code)]

#[cfg(not(windows))]
pub fn set_status(_message: &str) {}

#[cfg(not(windows))]
pub fn show_error(_message: &str) {}

#[cfg(not(windows))]
pub fn hand_over(_child_pid: u32) {}

#[cfg(windows)]
pub use imp::{hand_over, set_status, show_error};

#[cfg(windows)]
mod imp {
    use std::sync::Mutex;
    use std::thread;
    use std::time::{Duration, Instant};

    use once_cell::sync::Lazy;
    use windows_sys::Win32::Foundation::{
        CloseHandle, BOOL, COLORREF, HANDLE, HWND, LPARAM, LRESULT, RECT, WPARAM,
    };
    use windows_sys::Win32::Graphics::Gdi::{
        BeginPaint, CreateFontW, CreateSolidBrush, DeleteObject, DrawTextW, EndPaint, FillRect,
        InvalidateRect, SelectObject, SetBkMode, SetTextColor, DT_CENTER, DT_RIGHT, DT_SINGLELINE,
        DT_WORDBREAK, PAINTSTRUCT, TRANSPARENT,
    };
    use windows_sys::Win32::System::LibraryLoader::GetModuleHandleW;
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        CreateWindowExW, DefWindowProcW, DestroyWindow, DispatchMessageW, DrawIconEx,
        EnumWindows, GetMessageW, GetSystemMetrics, GetWindowThreadProcessId, IsWindowVisible,
        KillTimer, LoadImageW, MessageBoxW, PostMessageW, PostQuitMessage, RegisterClassW,
        SetTimer, ShowWindow, TranslateMessage, DI_NORMAL, IMAGE_ICON, MB_ICONERROR, MB_OK,
        MB_SETFOREGROUND, MB_TOPMOST, MSG, SM_CXSCREEN, SM_CYSCREEN, SW_SHOWNORMAL, WM_CLOSE,
        WM_DESTROY, WM_PAINT, WM_TIMER, WM_USER, WNDCLASSW, WS_EX_TOOLWINDOW, WS_EX_TOPMOST,
        WS_POPUP,
    };

    const WINDOW_WIDTH: i32 = 400;
    const WINDOW_HEIGHT: i32 = 200;
    const ICON_SIZE: i32 = 64;
    const BACKGROUND_COLOR: COLORREF = 0x00222222;
    const STATUS_COLOR: COLORREF = 0x00FFFFFF;
    const VERSION_COLOR: COLORREF = 0x00AAAAAA;
    const ELAPSED_HINT_AFTER_SECONDS: u64 = 5;
    const WM_STATUS_CHANGED: u32 = WM_USER + 1;
    const TICK_TIMER_ID: usize = 1;
    const HAND_OVER_TIMEOUT: Duration = Duration::from_secs(60);
    const HAND_OVER_POLL: Duration = Duration::from_millis(100);
    const PROCESS_QUERY_LIMITED_INFORMATION: u32 = 0x1000;
    const STILL_ACTIVE: u32 = 259;

    #[link(name = "kernel32")]
    extern "system" {
        fn OpenProcess(desired_access: u32, inherit_handle: BOOL, process_id: u32) -> HANDLE;
        fn GetExitCodeProcess(process: HANDLE, exit_code: *mut u32) -> BOOL;
    }

    struct SplashState {
        hwnd: Option<isize>,
        status: String,
        status_since: Instant,
        thread: Option<thread::JoinHandle<()>>,
    }

    static STATE: Lazy<Mutex<SplashState>> = Lazy::new(|| {
        Mutex::new(SplashState {
            hwnd: None,
            status: String::new(),
            status_since: Instant::now(),
            thread: None,
        })
    });

    fn wide(text: &str) -> Vec<u16> {
        text.encode_utf16().chain(std::iter::once(0)).collect()
    }

    fn status_line(message: &str) -> String {
        let mut line = message.trim().to_string();
        if let Some(first) = line.get(..1) {
            line = first.to_uppercase() + &line[1..];
        }
        if !line.ends_with("...") {
            line.push_str("...");
        }
        line
    }

    pub fn set_status(message: &str) {
        if !cfg!(pyapp_windows_subsystem) {
            return;
        }
        let mut state = STATE.lock().unwrap();
        state.status = status_line(message);
        state.status_since = Instant::now();
        if let Some(hwnd) = state.hwnd {
            unsafe { PostMessageW(hwnd as HWND, WM_STATUS_CHANGED, 0, 0) };
            return;
        }
        if state.thread.is_some() {
            return;
        }
        let (ready_sender, ready_receiver) = std::sync::mpsc::channel();
        state.thread = Some(thread::spawn(move || run_window(ready_sender)));
        drop(state);
        let _ = ready_receiver.recv_timeout(Duration::from_secs(5));
    }

    pub fn show_error(message: &str) {
        close();
        let title = wide("Whisper Key");
        let text = wide(&format!("Whisper Key could not be set up:\n\n{message}"));
        unsafe {
            MessageBoxW(
                std::ptr::null_mut(),
                text.as_ptr(),
                title.as_ptr(),
                MB_OK | MB_ICONERROR | MB_SETFOREGROUND | MB_TOPMOST,
            )
        };
    }

    pub fn hand_over(child_pid: u32) {
        if STATE.lock().unwrap().hwnd.is_none() {
            return;
        }
        set_status("Starting Whisper Key");
        let process = unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, child_pid) };
        let started = Instant::now();
        let mut exit_code = None;
        while started.elapsed() < HAND_OVER_TIMEOUT && !has_visible_window(child_pid) {
            exit_code = exited_code(process);
            if exit_code.is_some() {
                break;
            }
            thread::sleep(HAND_OVER_POLL);
        }
        if !process.is_null() {
            unsafe { CloseHandle(process) };
        }
        match exit_code {
            Some(code) if code != 0 => show_error(&format!(
                "The app stopped during startup (exit code {code}). See the log file for details."
            )),
            _ => close(),
        }
    }

    fn exited_code(process: HANDLE) -> Option<u32> {
        if process.is_null() {
            return None;
        }
        let mut code = STILL_ACTIVE;
        if unsafe { GetExitCodeProcess(process, &mut code) } == 0 || code == STILL_ACTIVE {
            return None;
        }
        Some(code)
    }

    fn close() {
        let (hwnd, handle) = {
            let mut state = STATE.lock().unwrap();
            (state.hwnd.take(), state.thread.take())
        };
        if let Some(hwnd) = hwnd {
            unsafe { PostMessageW(hwnd as HWND, WM_CLOSE, 0, 0) };
        }
        if let Some(handle) = handle {
            let _ = handle.join();
        }
    }

    struct WindowSearch {
        pid: u32,
        found: bool,
    }

    unsafe extern "system" fn find_window_of_process(hwnd: HWND, lparam: LPARAM) -> BOOL {
        let search = &mut *(lparam as *mut WindowSearch);
        let mut pid = 0u32;
        GetWindowThreadProcessId(hwnd, &mut pid);
        if pid == search.pid && IsWindowVisible(hwnd) != 0 {
            search.found = true;
            return 0;
        }
        1
    }

    fn has_visible_window(pid: u32) -> bool {
        let mut search = WindowSearch { pid, found: false };
        unsafe { EnumWindows(Some(find_window_of_process), &mut search as *mut _ as LPARAM) };
        search.found
    }

    fn run_window(ready: std::sync::mpsc::Sender<()>) {
        unsafe {
            let instance = GetModuleHandleW(std::ptr::null());
            let class_name = wide("WhisperKeySetupSplash");
            let class = WNDCLASSW {
                lpfnWndProc: Some(window_proc),
                hInstance: instance,
                lpszClassName: class_name.as_ptr(),
                hbrBackground: CreateSolidBrush(BACKGROUND_COLOR),
                ..std::mem::zeroed()
            };
            RegisterClassW(&class);
            let x = (GetSystemMetrics(SM_CXSCREEN) - WINDOW_WIDTH) / 2;
            let y = (GetSystemMetrics(SM_CYSCREEN) - WINDOW_HEIGHT) / 2;
            let title = wide("Whisper Key");
            let hwnd = CreateWindowExW(
                WS_EX_TOOLWINDOW | WS_EX_TOPMOST,
                class_name.as_ptr(),
                title.as_ptr(),
                WS_POPUP,
                x,
                y,
                WINDOW_WIDTH,
                WINDOW_HEIGHT,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                instance,
                std::ptr::null(),
            );
            if hwnd.is_null() {
                let _ = ready.send(());
                return;
            }
            STATE.lock().unwrap().hwnd = Some(hwnd as isize);
            ShowWindow(hwnd, SW_SHOWNORMAL);
            SetTimer(hwnd, TICK_TIMER_ID, 1000, None);
            let _ = ready.send(());

            let mut message: MSG = std::mem::zeroed();
            while GetMessageW(&mut message, std::ptr::null_mut(), 0, 0) > 0 {
                TranslateMessage(&message);
                DispatchMessageW(&message);
            }
        }
    }

    unsafe fn draw_text(hdc: windows_sys::Win32::Graphics::Gdi::HDC, text: &str, rect: &mut RECT,
                        color: COLORREF, size: i32, format: u32) {
        let face = wide("Segoe UI");
        let font = CreateFontW(-size, 0, 0, 0, 400, 0, 0, 0, 0, 0, 0, 5, 0, face.as_ptr());
        let previous = SelectObject(hdc, font);
        SetTextColor(hdc, color);
        let mut text = wide(text);
        DrawTextW(hdc, text.as_mut_ptr(), -1, rect, format);
        SelectObject(hdc, previous);
        DeleteObject(font);
    }

    unsafe fn paint(hwnd: HWND) {
        let mut paint_struct: PAINTSTRUCT = std::mem::zeroed();
        let hdc = BeginPaint(hwnd, &mut paint_struct);
        let mut full = RECT { left: 0, top: 0, right: WINDOW_WIDTH, bottom: WINDOW_HEIGHT };
        let brush = CreateSolidBrush(BACKGROUND_COLOR);
        FillRect(hdc, &full, brush);
        DeleteObject(brush);
        SetBkMode(hdc, TRANSPARENT as i32);

        let version = format!("Whisper Key {}", crate::app::project_version());
        let mut version_rect = RECT { left: 10, top: 6, right: WINDOW_WIDTH - 10, bottom: 24 };
        draw_text(hdc, &version, &mut version_rect, VERSION_COLOR, 11, DT_RIGHT | DT_SINGLELINE);

        let icon = LoadImageW(GetModuleHandleW(std::ptr::null()), 1 as _, IMAGE_ICON, ICON_SIZE, ICON_SIZE, 0);
        if !icon.is_null() {
            DrawIconEx(hdc, (WINDOW_WIDTH - ICON_SIZE) / 2, 40, icon as _, ICON_SIZE, ICON_SIZE, 0,
                       std::ptr::null_mut(), DI_NORMAL);
        }

        let (status, since) = {
            let state = STATE.lock().unwrap();
            (state.status.clone(), state.status_since)
        };
        let elapsed = since.elapsed().as_secs();
        let text = if elapsed < ELAPSED_HINT_AFTER_SECONDS { status } else { format!("{status} ({elapsed} s)") };
        full.top = 120;
        full.left = 10;
        full.right = WINDOW_WIDTH - 10;
        draw_text(hdc, &text, &mut full, STATUS_COLOR, 14, DT_CENTER | DT_WORDBREAK);
        EndPaint(hwnd, &paint_struct);
    }

    unsafe extern "system" fn window_proc(hwnd: HWND, message: u32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
        match message {
            WM_PAINT => {
                paint(hwnd);
                0
            }
            WM_TIMER | WM_STATUS_CHANGED => {
                InvalidateRect(hwnd, std::ptr::null(), 0);
                0
            }
            WM_CLOSE => {
                KillTimer(hwnd, TICK_TIMER_ID);
                DestroyWindow(hwnd);
                0
            }
            WM_DESTROY => {
                PostQuitMessage(0);
                0
            }
            _ => DefWindowProcW(hwnd, message, wparam, lparam),
        }
    }
}
