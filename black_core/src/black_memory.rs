use crate::black_error::{BlackError, BlackResult};
use parking_lot::Mutex;
use std::alloc::{self, Layout};
use std::collections::HashMap;

pub struct BlackMemoryPool {
    black_free_lists: Mutex<HashMap<usize, Vec<(*mut u8, Layout)>>>,
    black_min_bucket: usize,
    black_max_bucket: usize,
}

unsafe impl Send for BlackMemoryPool {}
unsafe impl Sync for BlackMemoryPool {}

impl BlackMemoryPool {
    pub fn black_new() -> Self {
        BlackMemoryPool {
            black_free_lists: Mutex::new(HashMap::new()),
            black_min_bucket: 64,
            black_max_bucket: 2 * 1024 * 1024 * 1024,
        }
    }

    fn black_bucket_size(&self, black_size: usize) -> usize {
        let mut black_bucket = self.black_min_bucket;
        while black_bucket < black_size && black_bucket < self.black_max_bucket {
            black_bucket *= 2;
        }
        black_bucket
    }

    pub fn black_alloc(&self, black_size: usize) -> BlackResult<(*mut u8, Layout)> {
        if black_size == 0 {
            let black_layout =
                Layout::from_size_align(0, 1).map_err(|black_e| BlackError::BlackMemoryError {
                    black_msg: format!("{}", black_e),
                })?;
            return Ok((std::ptr::null_mut(), black_layout));
        }

        let black_bucket = self.black_bucket_size(black_size);

        {
            let mut black_lists = self.black_free_lists.lock();
            if let Some(black_list) = black_lists.get_mut(&black_bucket) {
                if let Some(black_entry) = black_list.pop() {
                    return Ok(black_entry);
                }
            }
        }

        let black_layout = Layout::from_size_align(black_bucket, 64).map_err(|black_e| {
            BlackError::BlackMemoryError {
                black_msg: format!("{}", black_e),
            }
        })?;

        let black_ptr =
            unsafe { alloc::alloc_zeroed(black_layout) };
        if black_ptr.is_null() {
            return Err(BlackError::BlackMemoryError {
                black_msg: format!("failed to allocate {} bytes from pool", black_bucket),
            });
        }

        Ok((black_ptr, black_layout))
    }

    pub fn black_dealloc(&self, black_ptr: *mut u8, black_layout: Layout) {
        if black_ptr.is_null() {
            return;
        }
        let black_bucket = self.black_bucket_size(black_layout.size());
        let mut black_lists = self.black_free_lists.lock();
        black_lists
            .entry(black_bucket)
            .or_default()
            .push((black_ptr, black_layout));
    }

    pub fn black_clear(&self) {
        let mut black_lists = self.black_free_lists.lock();
        for (_, black_entries) in black_lists.drain() {
            for (black_ptr, black_layout) in black_entries {
                if !black_ptr.is_null() {
                    unsafe {
                        alloc::dealloc(black_ptr, black_layout);
                    }
                }
            }
        }
    }
}

impl Drop for BlackMemoryPool {
    fn drop(&mut self) {
        self.black_clear();
    }
}

pub struct BlackArena {
    black_buffer: *mut u8,
    black_capacity: usize,
    black_offset: Mutex<usize>,
    black_layout: Layout,
}

unsafe impl Send for BlackArena {}
unsafe impl Sync for BlackArena {}

impl BlackArena {
    pub fn black_new(black_capacity: usize) -> BlackResult<Self> {
        let black_layout =
            Layout::from_size_align(black_capacity, 64).map_err(|black_e| {
                BlackError::BlackMemoryError {
                    black_msg: format!("{}", black_e),
                }
            })?;

        let black_buffer =
            unsafe { alloc::alloc_zeroed(black_layout) };
        if black_buffer.is_null() {
            return Err(BlackError::BlackMemoryError {
                black_msg: format!("failed to allocate arena of {} bytes", black_capacity),
            });
        }

        Ok(BlackArena {
            black_buffer,
            black_capacity,
            black_offset: Mutex::new(0),
            black_layout,
        })
    }

    pub fn black_alloc(&self, black_size: usize, black_align: usize) -> BlackResult<*mut u8> {
        let mut black_off = self.black_offset.lock();
        let black_aligned = (*black_off + black_align - 1) & !(black_align - 1);

        if black_aligned + black_size > self.black_capacity {
            return Err(BlackError::BlackMemoryError {
                black_msg: "arena out of memory".into(),
            });
        }

        let black_ptr = unsafe { self.black_buffer.add(black_aligned) };
        *black_off = black_aligned + black_size;
        Ok(black_ptr)
    }

    pub fn black_reset(&self) {
        let mut black_off = self.black_offset.lock();
        *black_off = 0;
    }

    pub fn black_used(&self) -> usize {
        *self.black_offset.lock()
    }

    pub fn black_capacity(&self) -> usize {
        self.black_capacity
    }
}

impl Drop for BlackArena {
    fn drop(&mut self) {
        if !self.black_buffer.is_null() {
            unsafe {
                alloc::dealloc(self.black_buffer, self.black_layout);
            }
        }
    }
}
