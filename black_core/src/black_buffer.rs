use crate::black_device::BlackDevice;

use crate::black_error::{BlackError, BlackResult};
use std::alloc::{self, Layout};

#[derive(Debug)]
pub struct BlackBuffer {
    black_ptr: *mut u8,
    black_len: usize,
    black_layout: Layout,
    black_device: BlackDevice,
}

unsafe impl Send for BlackBuffer {}
unsafe impl Sync for BlackBuffer {}

impl BlackBuffer {
    pub fn black_alloc(black_size: usize, black_device: BlackDevice) -> BlackResult<Self> {
        if black_size == 0 {
            return Ok(BlackBuffer {
                black_ptr: std::ptr::null_mut(),
                black_len: 0,
                black_layout: Layout::from_size_align(0, 1).map_err(|black_e| {
                    BlackError::BlackMemoryError {
                        black_msg: format!("{}", black_e),
                    }
                })?,
                black_device,
            });
        }

        let black_align = 64;
        let black_layout =
            Layout::from_size_align(black_size, black_align).map_err(|black_e| {
                BlackError::BlackMemoryError {
                    black_msg: format!("{}", black_e),
                }
            })?;

        let black_ptr = match black_device {
            BlackDevice::BlackCpu => {
                let black_p = unsafe { alloc::alloc_zeroed(black_layout) };
                if black_p.is_null() {
                    return Err(BlackError::BlackMemoryError {
                        black_msg: format!("failed to allocate {} bytes", black_size),
                    });
                }
                black_p
            }
            BlackDevice::BlackCuda(_) => {
                return Err(BlackError::BlackDeviceError {
                    black_msg: "CUDA allocation requires black_cuda feature".into(),
                });
            }
            BlackDevice::BlackMetal(_) => {
                return Err(BlackError::BlackDeviceError {
                    black_msg: "Metal allocation not yet supported".into(),
                });
            }
        };

        Ok(BlackBuffer {
            black_ptr,
            black_len: black_size,
            black_layout,
            black_device,
        })
    }

    pub fn black_from_vec_f32(black_data: Vec<f32>) -> BlackResult<Self> {
        let black_byte_len = black_data.len() * std::mem::size_of::<f32>();
        let black_buf = Self::black_alloc(black_byte_len, BlackDevice::BlackCpu)?;
        if black_byte_len > 0 {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    black_data.as_ptr() as *const u8,
                    black_buf.black_ptr,
                    black_byte_len,
                );
            }
        }
        Ok(black_buf)
    }

    pub fn black_from_vec_f64(black_data: Vec<f64>) -> BlackResult<Self> {
        let black_byte_len = black_data.len() * std::mem::size_of::<f64>();
        let black_buf = Self::black_alloc(black_byte_len, BlackDevice::BlackCpu)?;
        if black_byte_len > 0 {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    black_data.as_ptr() as *const u8,
                    black_buf.black_ptr,
                    black_byte_len,
                );
            }
        }
        Ok(black_buf)
    }

    pub fn black_from_vec_i32(black_data: Vec<i32>) -> BlackResult<Self> {
        let black_byte_len = black_data.len() * std::mem::size_of::<i32>();
        let black_buf = Self::black_alloc(black_byte_len, BlackDevice::BlackCpu)?;
        if black_byte_len > 0 {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    black_data.as_ptr() as *const u8,
                    black_buf.black_ptr,
                    black_byte_len,
                );
            }
        }
        Ok(black_buf)
    }

    pub fn black_from_vec_i64(black_data: Vec<i64>) -> BlackResult<Self> {
        let black_byte_len = black_data.len() * std::mem::size_of::<i64>();
        let black_buf = Self::black_alloc(black_byte_len, BlackDevice::BlackCpu)?;
        if black_byte_len > 0 {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    black_data.as_ptr() as *const u8,
                    black_buf.black_ptr,
                    black_byte_len,
                );
            }
        }
        Ok(black_buf)
    }

    pub fn black_ptr(&self) -> *const u8 {
        self.black_ptr
    }

    pub fn black_mut_ptr(&mut self) -> *mut u8 {
        self.black_ptr
    }

    pub fn black_len(&self) -> usize {
        self.black_len
    }

    pub fn black_device(&self) -> &BlackDevice {
        &self.black_device
    }

    pub fn black_as_slice(&self) -> &[u8] {
        if self.black_len == 0 {
            return &[];
        }
        unsafe { std::slice::from_raw_parts(self.black_ptr, self.black_len) }
    }

    pub fn black_as_mut_slice(&mut self) -> &mut [u8] {
        if self.black_len == 0 {
            return &mut [];
        }
        unsafe { std::slice::from_raw_parts_mut(self.black_ptr, self.black_len) }
    }

    pub fn black_as_f32_slice(&self) -> &[f32] {
        if self.black_len == 0 {
            return &[];
        }
        let black_count = self.black_len / std::mem::size_of::<f32>();
        unsafe { std::slice::from_raw_parts(self.black_ptr as *const f32, black_count) }
    }

    pub fn black_as_f32_mut_slice(&mut self) -> &mut [f32] {
        if self.black_len == 0 {
            return &mut [];
        }
        let black_count = self.black_len / std::mem::size_of::<f32>();
        unsafe { std::slice::from_raw_parts_mut(self.black_ptr as *mut f32, black_count) }
    }

    pub fn black_as_f64_slice(&self) -> &[f64] {
        if self.black_len == 0 {
            return &[];
        }
        let black_count = self.black_len / std::mem::size_of::<f64>();
        unsafe { std::slice::from_raw_parts(self.black_ptr as *const f64, black_count) }
    }

    pub fn black_as_f64_mut_slice(&mut self) -> &mut [f64] {
        if self.black_len == 0 {
            return &mut [];
        }
        let black_count = self.black_len / std::mem::size_of::<f64>();
        unsafe { std::slice::from_raw_parts_mut(self.black_ptr as *mut f64, black_count) }
    }

    pub fn black_copy_from(&mut self, black_other: &BlackBuffer) -> BlackResult<()> {
        if self.black_len != black_other.black_len {
            return Err(BlackError::BlackMemoryError {
                black_msg: "buffer size mismatch in copy".into(),
            });
        }
        if self.black_len > 0 {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    black_other.black_ptr,
                    self.black_ptr,
                    self.black_len,
                );
            }
        }
        Ok(())
    }

    pub fn black_zero(&mut self) {
        if self.black_len > 0 {
            unsafe {
                std::ptr::write_bytes(self.black_ptr, 0, self.black_len);
            }
        }
    }
}

impl Drop for BlackBuffer {
    fn drop(&mut self) {
        if !self.black_ptr.is_null() && self.black_len > 0 {
            match self.black_device {
                BlackDevice::BlackCpu => unsafe {
                    alloc::dealloc(self.black_ptr, self.black_layout);
                },
                _ => {}
            }
        }
    }
}

impl Clone for BlackBuffer {
    fn clone(&self) -> Self {
        if self.black_len == 0 {
            return BlackBuffer {
                black_ptr: std::ptr::null_mut(),
                black_len: 0,
                black_layout: self.black_layout,
                black_device: self.black_device,
            };
        }

        let black_ptr = unsafe { alloc::alloc(self.black_layout) };
        if !black_ptr.is_null() {
            unsafe {
                std::ptr::copy_nonoverlapping(self.black_ptr, black_ptr, self.black_len);
            }
        }
        BlackBuffer {
            black_ptr,
            black_len: self.black_len,
            black_layout: self.black_layout,
            black_device: self.black_device,
        }
    }
}
