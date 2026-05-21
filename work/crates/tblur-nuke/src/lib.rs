unsafe extern "C" {
    fn tblur_base_keepalive();
}

#[unsafe(no_mangle)]
pub extern "C" fn tblur_base_rust_link() {
    unsafe {
        tblur_base_keepalive();
    }
}
