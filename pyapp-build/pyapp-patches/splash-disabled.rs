pub fn set_status(_message: &str) {}

pub fn hand_over(_child_pid: u32) {}

pub fn prepare_child(_command: &mut std::process::Command) {}

#[cfg(not(windows))]
pub fn show_error(_message: &str) {}

#[cfg(windows)]
pub fn show_error(message: &str) {
    use windows_sys::Win32::UI::WindowsAndMessaging::{MessageBoxW, MB_ICONERROR, MB_OK, MB_SETFOREGROUND};
    let text: Vec<u16> = message.encode_utf16().chain(std::iter::once(0)).collect();
    let title: Vec<u16> = "Whisper Key".encode_utf16().chain(std::iter::once(0)).collect();
    unsafe {
        MessageBoxW(std::ptr::null_mut(), text.as_ptr(), title.as_ptr(), MB_OK | MB_ICONERROR | MB_SETFOREGROUND);
    }
}
