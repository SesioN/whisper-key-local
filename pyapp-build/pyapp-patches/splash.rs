#![allow(dead_code)]

#[cfg(not(windows))]
pub fn set_status(_message: &str) {}

#[cfg(not(windows))]
pub fn show_error(_message: &str) {}

#[cfg(not(windows))]
pub fn hand_over(_child_pid: u32) {}

#[cfg(not(windows))]
pub fn prepare_child(_command: &mut std::process::Command) {}

#[cfg(windows)]
pub use imp::{hand_over, prepare_child, set_status, show_error};

#[cfg(windows)]
mod imp {
    use std::sync::Mutex;
    use std::thread;
    use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

    use once_cell::sync::Lazy;
    use windows_sys::Win32::Foundation::{
        CloseHandle, BOOL, COLORREF, HANDLE, HWND, INVALID_HANDLE_VALUE, LPARAM, LRESULT, RECT, WPARAM,
    };
    use windows_sys::Win32::Graphics::Dwm::{DwmSetWindowAttribute, DWMWA_WINDOW_CORNER_PREFERENCE, DWMWCP_ROUND};
    use windows_sys::Win32::Graphics::Gdi::{
        BeginPaint, BitBlt, CreateCompatibleBitmap, CreateCompatibleDC, CreateDIBSection, CreateFontW, DeleteDC,
        DeleteObject, DrawTextW, EndPaint, InvalidateRect, SelectObject, SetBkMode, SetTextColor, BITMAPINFO,
        BITMAPINFOHEADER, BI_RGB, DIB_RGB_COLORS, DT_CALCRECT, DT_CENTER, DT_RIGHT, DT_SINGLELINE, DT_WORDBREAK,
        HBITMAP, HDC, PAINTSTRUCT, SRCCOPY, TRANSPARENT,
    };
    use windows_sys::Win32::System::Diagnostics::ToolHelp::{
        CreateToolhelp32Snapshot, Process32FirstW, Process32NextW, PROCESSENTRY32W, TH32CS_SNAPPROCESS,
    };
    use windows_sys::Win32::System::LibraryLoader::GetModuleHandleW;
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        CreateWindowExW, DefWindowProcW, DestroyWindow, DispatchMessageW, EnumWindows, GetMessageW,
        GetSystemMetrics, GetWindowThreadProcessId, IsWindowVisible, KillTimer, MessageBoxW, PostMessageW,
        PostQuitMessage, RegisterClassW, SetTimer, ShowWindow, TranslateMessage, MB_ICONERROR, MB_OK,
        MB_SETFOREGROUND, MB_TOPMOST, MSG, SM_CXSCREEN, SM_CYSCREEN, SW_SHOWNORMAL, WM_CLOSE, WM_DESTROY,
        WM_ERASEBKGND, WM_PAINT, WM_TIMER, WM_USER, WNDCLASSW, WS_EX_TOOLWINDOW, WS_EX_TOPMOST, WS_POPUP,
    };

    const WINDOW_WIDTH: i32 = 440;
    const WINDOW_HEIGHT: i32 = 260;
    const FRAME_COUNT: i32 = 40;
    const FRAME_INTERVAL_MS: u32 = 77;
    const SPLASH_FRAMES_JPEG: &[u8] = include_bytes!("splash_frames.jpg");
    const STATUS_CENTER_Y: i32 = 199;
    const STATUS_WRAP_WIDTH: i32 = 368;
    const STATUS_COLOR: COLORREF = 0x00FFF5F2;
    const VERSION_COLOR: COLORREF = 0x00D6A69A;
    const VERSION_MARGIN: i32 = 10;
    const STATUS_FONT_HEIGHT: i32 = 15;
    const VERSION_FONT_HEIGHT: i32 = 11;
    const ELAPSED_HINT_AFTER_SECONDS: u64 = 5;
    const SPLASH_EPOCH_VARIABLE: &str = "WHISPER_KEY_SPLASH_EPOCH_MS";
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
        animation_epoch: SystemTime,
        frames: Option<isize>,
        thread: Option<thread::JoinHandle<()>>,
    }

    static STATE: Lazy<Mutex<SplashState>> = Lazy::new(|| {
        Mutex::new(SplashState {
            hwnd: None,
            status: String::new(),
            status_since: Instant::now(),
            animation_epoch: SystemTime::now(),
            frames: None,
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

    pub fn prepare_child(command: &mut std::process::Command) {
        let state = STATE.lock().unwrap();
        if state.hwnd.is_none() {
            return;
        }
        if let Ok(epoch) = state.animation_epoch.duration_since(UNIX_EPOCH) {
            command.env(SPLASH_EPOCH_VARIABLE, epoch.as_millis().to_string());
        }
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
        pids: Vec<u32>,
        found: bool,
    }

    unsafe extern "system" fn find_window_of_process(hwnd: HWND, lparam: LPARAM) -> BOOL {
        let search = &mut *(lparam as *mut WindowSearch);
        let mut pid = 0u32;
        GetWindowThreadProcessId(hwnd, &mut pid);
        if search.pids.contains(&pid) && IsWindowVisible(hwnd) != 0 {
            search.found = true;
            return 0;
        }
        1
    }

    fn parent_links() -> Vec<(u32, u32)> {
        let mut links = Vec::new();
        unsafe {
            let snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
            if snapshot == INVALID_HANDLE_VALUE {
                return links;
            }
            let mut entry: PROCESSENTRY32W = std::mem::zeroed();
            entry.dwSize = std::mem::size_of::<PROCESSENTRY32W>() as u32;
            let mut has_entry = Process32FirstW(snapshot, &mut entry) != 0;
            while has_entry {
                links.push((entry.th32ProcessID, entry.th32ParentProcessID));
                has_entry = Process32NextW(snapshot, &mut entry) != 0;
            }
            CloseHandle(snapshot);
        }
        links
    }

    fn process_tree(root_pid: u32) -> Vec<u32> {
        let links = parent_links();
        let mut pids = vec![root_pid];
        let mut index = 0;
        while index < pids.len() {
            let parent = pids[index];
            for &(pid, parent_pid) in &links {
                if parent_pid == parent && pid != parent && !pids.contains(&pid) {
                    pids.push(pid);
                }
            }
            index += 1;
        }
        pids
    }

    fn has_visible_window(root_pid: u32) -> bool {
        let mut search = WindowSearch { pids: process_tree(root_pid), found: false };
        unsafe { EnumWindows(Some(find_window_of_process), &mut search as *mut _ as LPARAM) };
        search.found
    }

    fn decode_frames() -> Option<HBITMAP> {
        let mut decoder = jpeg_decoder::Decoder::new(SPLASH_FRAMES_JPEG);
        let pixels = decoder.decode().ok()?;
        let info = decoder.info()?;
        let (width, height) = (info.width as i32, info.height as i32);
        if info.pixel_format != jpeg_decoder::PixelFormat::RGB24
            || width != WINDOW_WIDTH * FRAME_COUNT
            || height != WINDOW_HEIGHT
        {
            return None;
        }
        unsafe {
            let mut header: BITMAPINFO = std::mem::zeroed();
            header.bmiHeader = BITMAPINFOHEADER {
                biSize: std::mem::size_of::<BITMAPINFOHEADER>() as u32,
                biWidth: width,
                biHeight: -height,
                biPlanes: 1,
                biBitCount: 32,
                biCompression: BI_RGB,
                ..std::mem::zeroed()
            };
            let mut bits: *mut std::ffi::c_void = std::ptr::null_mut();
            let bitmap = CreateDIBSection(std::ptr::null_mut(), &header, DIB_RGB_COLORS, &mut bits,
                                          std::ptr::null_mut(), 0);
            if bitmap.is_null() || bits.is_null() {
                return None;
            }
            let target = std::slice::from_raw_parts_mut(bits as *mut u8, (width * height * 4) as usize);
            for (source, destination) in pixels.chunks_exact(3).zip(target.chunks_exact_mut(4)) {
                destination[0] = source[2];
                destination[1] = source[1];
                destination[2] = source[0];
                destination[3] = 255;
            }
            Some(bitmap)
        }
    }

    fn current_frame(epoch: SystemTime) -> i32 {
        let elapsed = SystemTime::now().duration_since(epoch).unwrap_or_default().as_millis();
        ((elapsed / FRAME_INTERVAL_MS as u128) % FRAME_COUNT as u128) as i32
    }

    fn run_window(ready: std::sync::mpsc::Sender<()>) {
        unsafe {
            let instance = GetModuleHandleW(std::ptr::null());
            let class_name = wide("WhisperKeySetupSplash");
            let class = WNDCLASSW {
                lpfnWndProc: Some(window_proc),
                hInstance: instance,
                lpszClassName: class_name.as_ptr(),
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
            let corner = DWMWCP_ROUND;
            DwmSetWindowAttribute(
                hwnd,
                DWMWA_WINDOW_CORNER_PREFERENCE as u32,
                &corner as *const _ as *const std::ffi::c_void,
                std::mem::size_of_val(&corner) as u32,
            );
            let frames = decode_frames();
            {
                let mut state = STATE.lock().unwrap();
                state.hwnd = Some(hwnd as isize);
                state.frames = frames.map(|bitmap| bitmap as isize);
                state.animation_epoch = SystemTime::now();
            }
            ShowWindow(hwnd, SW_SHOWNORMAL);
            SetTimer(hwnd, TICK_TIMER_ID, FRAME_INTERVAL_MS, None);
            let _ = ready.send(());

            let mut message: MSG = std::mem::zeroed();
            while GetMessageW(&mut message, std::ptr::null_mut(), 0, 0) > 0 {
                TranslateMessage(&message);
                DispatchMessageW(&message);
            }
            if let Some(bitmap) = STATE.lock().unwrap().frames.take() {
                DeleteObject(bitmap as _);
            }
        }
    }

    unsafe fn draw_text(hdc: HDC, text: &str, rect: &mut RECT, color: COLORREF, height: i32, format: u32) {
        let face = wide("Segoe UI");
        let font = CreateFontW(-height, 0, 0, 0, 400, 0, 0, 0, 0, 0, 0, 5, 0, face.as_ptr());
        let previous = SelectObject(hdc, font);
        SetTextColor(hdc, color);
        let mut text = wide(text);
        DrawTextW(hdc, text.as_mut_ptr(), -1, rect, format);
        SelectObject(hdc, previous);
        DeleteObject(font);
    }

    unsafe fn paint(hwnd: HWND) {
        let mut paint_struct: PAINTSTRUCT = std::mem::zeroed();
        let window_dc = BeginPaint(hwnd, &mut paint_struct);
        let buffer_dc = CreateCompatibleDC(window_dc);
        let buffer = CreateCompatibleBitmap(window_dc, WINDOW_WIDTH, WINDOW_HEIGHT);
        let previous_buffer = SelectObject(buffer_dc, buffer);

        let (status, since, epoch, frames) = {
            let state = STATE.lock().unwrap();
            (state.status.clone(), state.status_since, state.animation_epoch, state.frames)
        };
        if let Some(frames) = frames {
            let frames_dc = CreateCompatibleDC(window_dc);
            let previous_frames = SelectObject(frames_dc, frames as _);
            BitBlt(buffer_dc, 0, 0, WINDOW_WIDTH, WINDOW_HEIGHT, frames_dc, current_frame(epoch) * WINDOW_WIDTH, 0,
                   SRCCOPY);
            SelectObject(frames_dc, previous_frames);
            DeleteDC(frames_dc);
        }
        SetBkMode(buffer_dc, TRANSPARENT as i32);

        let version = crate::app::project_version();
        let mut version_rect = RECT {
            left: VERSION_MARGIN,
            top: VERSION_MARGIN - 4,
            right: WINDOW_WIDTH - VERSION_MARGIN,
            bottom: VERSION_MARGIN + 16,
        };
        draw_text(buffer_dc, &version, &mut version_rect, VERSION_COLOR, VERSION_FONT_HEIGHT, DT_RIGHT | DT_SINGLELINE);

        let elapsed = since.elapsed().as_secs();
        let text = if elapsed < ELAPSED_HINT_AFTER_SECONDS { status } else { format!("{status} ({elapsed} s)") };
        let left = (WINDOW_WIDTH - STATUS_WRAP_WIDTH) / 2;
        let mut measured = RECT { left, top: 0, right: left + STATUS_WRAP_WIDTH, bottom: 0 };
        draw_text(buffer_dc, &text, &mut measured, STATUS_COLOR, STATUS_FONT_HEIGHT,
                  DT_CENTER | DT_WORDBREAK | DT_CALCRECT);
        let text_height = measured.bottom - measured.top;
        let mut status_rect = RECT {
            left,
            top: STATUS_CENTER_Y - text_height / 2,
            right: left + STATUS_WRAP_WIDTH,
            bottom: STATUS_CENTER_Y + text_height - text_height / 2,
        };
        draw_text(buffer_dc, &text, &mut status_rect, STATUS_COLOR, STATUS_FONT_HEIGHT, DT_CENTER | DT_WORDBREAK);

        BitBlt(window_dc, 0, 0, WINDOW_WIDTH, WINDOW_HEIGHT, buffer_dc, 0, 0, SRCCOPY);
        SelectObject(buffer_dc, previous_buffer);
        DeleteObject(buffer);
        DeleteDC(buffer_dc);
        EndPaint(hwnd, &paint_struct);
    }

    unsafe extern "system" fn window_proc(hwnd: HWND, message: u32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
        match message {
            WM_PAINT => {
                paint(hwnd);
                0
            }
            WM_ERASEBKGND => 1,
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
