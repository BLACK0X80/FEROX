use crate::black_buffer::BlackBuffer;
use crate::black_device::BlackDevice;
use crate::black_dtype::BlackDType;
use crate::black_error::{BlackError, BlackResult};
use crate::black_shape::BlackShape;
use crate::black_strides::BlackStrides;
use rand::Rng;
use rand_distr::StandardNormal;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct BlackTensor {
    pub black_buffer: Arc<BlackBuffer>,
    pub black_shape: BlackShape,
    pub black_strides: BlackStrides,
    pub black_dtype: BlackDType,
    pub black_device: BlackDevice,
    pub black_offset: usize,
}

impl BlackTensor {
    pub fn black_new(
        black_buffer: BlackBuffer,
        black_shape: BlackShape,
        black_dtype: BlackDType,
        black_device: BlackDevice,
    ) -> Self {
        let black_strides = BlackStrides::black_from_shape(&black_shape);
        BlackTensor {
            black_buffer: Arc::new(black_buffer),
            black_shape,
            black_strides,
            black_dtype,
            black_device,
            black_offset: 0,
        }
    }

    pub fn black_zeros(black_shape: &[usize], black_dtype: BlackDType) -> BlackResult<Self> {
        let black_s = BlackShape::black_new(black_shape);
        let black_numel = black_s.black_numel();
        let black_byte_len = black_numel * black_dtype.black_size_in_bytes();
        let black_buf = BlackBuffer::black_alloc(black_byte_len, BlackDevice::BlackCpu)?;
        Ok(Self::black_new(
            black_buf,
            black_s,
            black_dtype,
            BlackDevice::BlackCpu,
        ))
    }

    pub fn black_ones(black_shape: &[usize], black_dtype: BlackDType) -> BlackResult<Self> {
        let black_s = BlackShape::black_new(black_shape);
        let black_numel = black_s.black_numel();
        match black_dtype {
            BlackDType::BlackF32 => {
                let black_data = vec![1.0f32; black_numel];
                let black_buf = BlackBuffer::black_from_vec_f32(black_data)?;
                Ok(Self::black_new(
                    black_buf,
                    black_s,
                    black_dtype,
                    BlackDevice::BlackCpu,
                ))
            }
            BlackDType::BlackF64 => {
                let black_data = vec![1.0f64; black_numel];
                let black_buf = BlackBuffer::black_from_vec_f64(black_data)?;
                Ok(Self::black_new(
                    black_buf,
                    black_s,
                    black_dtype,
                    BlackDevice::BlackCpu,
                ))
            }
            BlackDType::BlackI32 => {
                let black_data = vec![1i32; black_numel];
                let black_buf = BlackBuffer::black_from_vec_i32(black_data)?;
                Ok(Self::black_new(
                    black_buf,
                    black_s,
                    black_dtype,
                    BlackDevice::BlackCpu,
                ))
            }
            _ => {
                let black_byte_len = black_numel * black_dtype.black_size_in_bytes();
                let mut black_buf =
                    BlackBuffer::black_alloc(black_byte_len, BlackDevice::BlackCpu)?;
                let black_slice = black_buf.black_as_mut_slice();
                for black_val in black_slice.iter_mut().take(black_byte_len) {
                    *black_val = 1;
                }
                Ok(Self::black_new(
                    black_buf,
                    black_s,
                    black_dtype,
                    BlackDevice::BlackCpu,
                ))
            }
        }
    }

    pub fn black_rand(black_shape: &[usize]) -> BlackResult<Self> {
        let black_s = BlackShape::black_new(black_shape);
        let black_numel = black_s.black_numel();
        let mut black_rng = rand::thread_rng();
        let black_data: Vec<f32> = (0..black_numel).map(|_| black_rng.gen::<f32>()).collect();
        let black_buf = BlackBuffer::black_from_vec_f32(black_data)?;
        Ok(Self::black_new(
            black_buf,
            black_s,
            BlackDType::BlackF32,
            BlackDevice::BlackCpu,
        ))
    }

    pub fn black_randn(black_shape: &[usize]) -> BlackResult<Self> {
        let black_s = BlackShape::black_new(black_shape);
        let black_numel = black_s.black_numel();
        let mut black_rng = rand::thread_rng();
        let black_data: Vec<f32> = (0..black_numel)
            .map(|_| black_rng.sample::<f32, _>(StandardNormal))
            .collect();
        let black_buf = BlackBuffer::black_from_vec_f32(black_data)?;
        Ok(Self::black_new(
            black_buf,
            black_s,
            BlackDType::BlackF32,
            BlackDevice::BlackCpu,
        ))
    }

    pub fn black_from_slice_f32(black_data: &[f32], black_shape: &[usize]) -> BlackResult<Self> {
        let black_s = BlackShape::black_new(black_shape);
        if black_s.black_numel() != black_data.len() {
            return Err(BlackError::BlackShapeError {
                black_msg: format!(
                    "data length {} does not match shape {:?}",
                    black_data.len(),
                    black_shape
                ),
            });
        }
        let black_buf = BlackBuffer::black_from_vec_f32(black_data.to_vec())?;
        Ok(Self::black_new(
            black_buf,
            black_s,
            BlackDType::BlackF32,
            BlackDevice::BlackCpu,
        ))
    }

    pub fn black_from_slice_f64(black_data: &[f64], black_shape: &[usize]) -> BlackResult<Self> {
        let black_s = BlackShape::black_new(black_shape);
        if black_s.black_numel() != black_data.len() {
            return Err(BlackError::BlackShapeError {
                black_msg: format!(
                    "data length {} does not match shape {:?}",
                    black_data.len(),
                    black_shape
                ),
            });
        }
        let black_buf = BlackBuffer::black_from_vec_f64(black_data.to_vec())?;
        Ok(Self::black_new(
            black_buf,
            black_s,
            BlackDType::BlackF64,
            BlackDevice::BlackCpu,
        ))
    }

    pub fn black_scalar_f32(black_val: f32) -> BlackResult<Self> {
        let black_buf = BlackBuffer::black_from_vec_f32(vec![black_val])?;
        Ok(Self::black_new(
            black_buf,
            BlackShape::black_scalar(),
            BlackDType::BlackF32,
            BlackDevice::BlackCpu,
        ))
    }

    pub fn black_arange(black_start: f32, black_end: f32, black_step: f32) -> BlackResult<Self> {
        let mut black_data = Vec::new();
        let mut black_val = black_start;
        while black_val < black_end {
            black_data.push(black_val);
            black_val += black_step;
        }
        let black_len = black_data.len();
        let black_buf = BlackBuffer::black_from_vec_f32(black_data)?;
        Ok(Self::black_new(
            black_buf,
            BlackShape::black_new(&[black_len]),
            BlackDType::BlackF32,
            BlackDevice::BlackCpu,
        ))
    }

    pub fn black_full(
        black_shape: &[usize],
        black_val: f32,
        black_dtype: BlackDType,
    ) -> BlackResult<Self> {
        let black_s = BlackShape::black_new(black_shape);
        let black_numel = black_s.black_numel();
        let black_data = vec![black_val; black_numel];
        let black_buf = BlackBuffer::black_from_vec_f32(black_data)?;
        Ok(Self::black_new(
            black_buf,
            black_s,
            black_dtype,
            BlackDevice::BlackCpu,
        ))
    }

    pub fn black_numel(&self) -> usize {
        self.black_shape.black_numel()
    }

    pub fn black_ndim(&self) -> usize {
        self.black_shape.black_ndim()
    }

    pub fn black_is_contiguous(&self) -> bool {
        self.black_strides.black_is_contiguous(&self.black_shape)
    }

    pub fn black_contiguous(&self) -> BlackResult<Self> {
        if self.black_is_contiguous() {
            return Ok(self.clone());
        }

        let black_numel = self.black_numel();
        let black_byte_len = black_numel * self.black_dtype.black_size_in_bytes();
        let mut black_new_buf = BlackBuffer::black_alloc(black_byte_len, self.black_device)?;

        let black_src = self.black_buffer.black_as_f32_slice();
        let black_dst = black_new_buf.black_as_f32_mut_slice();
        let black_dims = self.black_shape.black_dims();
        let black_strides_sl = self.black_strides.black_as_slice();

        let mut black_indices = vec![0usize; black_dims.len()];
        for black_dst_val in black_dst.iter_mut().take(black_numel) {
            let mut black_src_offset = self.black_offset;
            for black_d in 0..black_dims.len() {
                black_src_offset += black_indices[black_d] * black_strides_sl[black_d];
            }
            *black_dst_val = black_src[black_src_offset];

            let mut black_carry = true;
            for black_d in (0..black_dims.len()).rev() {
                if black_carry {
                    black_indices[black_d] += 1;
                    if black_indices[black_d] < black_dims[black_d] {
                        black_carry = false;
                    } else {
                        black_indices[black_d] = 0;
                    }
                }
            }
        }

        let black_new_strides = BlackStrides::black_from_shape(&self.black_shape);
        Ok(BlackTensor {
            black_buffer: Arc::new(black_new_buf),
            black_shape: self.black_shape.clone(),
            black_strides: black_new_strides,
            black_dtype: self.black_dtype,
            black_device: self.black_device,
            black_offset: 0,
        })
    }

    pub fn black_view(&self, black_new_shape: &[usize]) -> BlackResult<Self> {
        let black_new_s = BlackShape::black_new(black_new_shape);
        if black_new_s.black_numel() != self.black_numel() {
            return Err(BlackError::BlackShapeError {
                black_msg: format!(
                    "cannot view tensor of {} elements as shape {:?}",
                    self.black_numel(),
                    black_new_shape
                ),
            });
        }
        if !self.black_is_contiguous() {
            return Err(BlackError::BlackShapeError {
                black_msg: "cannot view non-contiguous tensor".into(),
            });
        }

        let black_new_strides = BlackStrides::black_from_shape(&black_new_s);
        Ok(BlackTensor {
            black_buffer: Arc::clone(&self.black_buffer),
            black_shape: black_new_s,
            black_strides: black_new_strides,
            black_dtype: self.black_dtype,
            black_device: self.black_device,
            black_offset: self.black_offset,
        })
    }

    pub fn black_reshape(&self, black_new_shape: &[usize]) -> BlackResult<Self> {
        if self.black_is_contiguous() {
            self.black_view(black_new_shape)
        } else {
            let black_contig = self.black_contiguous()?;
            black_contig.black_view(black_new_shape)
        }
    }

    pub fn black_transpose(&self, black_dim0: usize, black_dim1: usize) -> BlackResult<Self> {
        let black_ndim = self.black_ndim();
        if black_dim0 >= black_ndim || black_dim1 >= black_ndim {
            return Err(BlackError::BlackIndexError {
                black_msg: format!(
                    "transpose dims ({}, {}) out of range for {}D tensor",
                    black_dim0, black_dim1, black_ndim
                ),
            });
        }

        let mut black_new_dims: Vec<usize> = self.black_shape.black_dims().to_vec();
        black_new_dims.swap(black_dim0, black_dim1);

        let mut black_new_strides: Vec<usize> = self.black_strides.black_as_slice().to_vec();
        black_new_strides.swap(black_dim0, black_dim1);

        Ok(BlackTensor {
            black_buffer: Arc::clone(&self.black_buffer),
            black_shape: BlackShape::black_new(&black_new_dims),
            black_strides: BlackStrides::black_new(black_new_strides),
            black_dtype: self.black_dtype,
            black_device: self.black_device,
            black_offset: self.black_offset,
        })
    }

    pub fn black_t(&self) -> BlackResult<Self> {
        let black_ndim = self.black_ndim();
        if black_ndim < 2 {
            return Err(BlackError::BlackShapeError {
                black_msg: "cannot transpose tensor with fewer than 2 dimensions".into(),
            });
        }
        self.black_transpose(black_ndim - 2, black_ndim - 1)
    }

    pub fn black_permute(&self, black_dims: &[usize]) -> BlackResult<Self> {
        let black_ndim = self.black_ndim();
        if black_dims.len() != black_ndim {
            return Err(BlackError::BlackShapeError {
                black_msg: format!(
                    "permute dims length {} does not match tensor rank {}",
                    black_dims.len(),
                    black_ndim
                ),
            });
        }

        let black_old_dims = self.black_shape.black_dims();
        let black_old_strides = self.black_strides.black_as_slice();
        let mut black_new_dims = vec![0usize; black_ndim];
        let mut black_new_strides = vec![0usize; black_ndim];

        for black_i in 0..black_ndim {
            black_new_dims[black_i] = black_old_dims[black_dims[black_i]];
            black_new_strides[black_i] = black_old_strides[black_dims[black_i]];
        }

        Ok(BlackTensor {
            black_buffer: Arc::clone(&self.black_buffer),
            black_shape: BlackShape::black_new(&black_new_dims),
            black_strides: BlackStrides::black_new(black_new_strides),
            black_dtype: self.black_dtype,
            black_device: self.black_device,
            black_offset: self.black_offset,
        })
    }

    pub fn black_as_f32_slice(&self) -> BlackResult<&[f32]> {
        if self.black_dtype != BlackDType::BlackF32 {
            return Err(BlackError::BlackDTypeError {
                black_msg: "tensor is not f32".into(),
            });
        }
        Ok(self.black_buffer.black_as_f32_slice())
    }

    pub fn black_to_vec_f32(&self) -> BlackResult<Vec<f32>> {
        let black_contig = self.black_contiguous()?;
        match black_contig.black_dtype {
            BlackDType::BlackF32 => {
                let black_slice = black_contig.black_buffer.black_as_f32_slice();
                Ok(black_slice[black_contig.black_offset
                    ..black_contig.black_offset + black_contig.black_numel()]
                    .to_vec())
            }
            _ => Err(BlackError::BlackDTypeError {
                black_msg: "cannot convert non-f32 tensor to Vec<f32>".into(),
            }),
        }
    }

    pub fn black_item_f32(&self) -> BlackResult<f32> {
        if self.black_numel() != 1 {
            return Err(BlackError::BlackShapeError {
                black_msg: "item() requires a single-element tensor".into(),
            });
        }
        let black_data = self.black_to_vec_f32()?;
        Ok(black_data[0])
    }

    pub fn black_norm(&self) -> BlackResult<f32> {
        let black_data = self.black_to_vec_f32()?;
        let black_sum: f32 = black_data.iter().map(|black_v| black_v * black_v).sum();
        Ok(black_sum.sqrt())
    }

    pub fn black_scale_inplace(&mut self, black_factor: f32) -> BlackResult<()> {
        let black_buf = Arc::make_mut(&mut self.black_buffer);
        let black_slice = black_buf.black_as_f32_mut_slice();
        for black_v in black_slice.iter_mut() {
            *black_v *= black_factor;
        }
        Ok(())
    }

    pub fn black_is_finite(&self) -> BlackResult<bool> {
        let black_data = self.black_to_vec_f32()?;
        Ok(black_data.iter().all(|black_v| black_v.is_finite()))
    }
}

impl std::fmt::Display for BlackTensor {
    fn fmt(&self, black_f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            black_f,
            "BlackTensor(shape={}, dtype={}, device={})",
            self.black_shape, self.black_dtype, self.black_device
        )
    }
}
