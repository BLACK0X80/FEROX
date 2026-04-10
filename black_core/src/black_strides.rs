use crate::black_shape::BlackShape;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlackStrides {
    black_data: Vec<usize>,
}

impl BlackStrides {
    pub fn black_from_shape(black_shape: &BlackShape) -> Self {
        let black_dims = black_shape.black_dims();
        let black_rank = black_dims.len();
        if black_rank == 0 {
            return BlackStrides {
                black_data: vec![],
            };
        }

        let mut black_strides = vec![0usize; black_rank];
        black_strides[black_rank - 1] = 1;
        for black_i in (0..black_rank - 1).rev() {
            black_strides[black_i] = black_strides[black_i + 1] * black_dims[black_i + 1];
        }

        BlackStrides {
            black_data: black_strides,
        }
    }

    pub fn black_new(black_strides: Vec<usize>) -> Self {
        BlackStrides {
            black_data: black_strides,
        }
    }

    pub fn black_as_slice(&self) -> &[usize] {
        &self.black_data
    }

    pub fn black_is_contiguous(&self, black_shape: &BlackShape) -> bool {
        let black_expected = BlackStrides::black_from_shape(black_shape);
        self.black_data == black_expected.black_data
    }

    pub fn black_offset(&self, black_indices: &[usize]) -> usize {
        let mut black_offset = 0usize;
        for (black_i, black_idx) in black_indices.iter().enumerate() {
            if black_i < self.black_data.len() {
                black_offset += black_idx * self.black_data[black_i];
            }
        }
        black_offset
    }

    pub fn black_ndim(&self) -> usize {
        self.black_data.len()
    }
}

impl std::fmt::Display for BlackStrides {
    fn fmt(&self, black_f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(black_f, "BlackStrides({:?})", self.black_data)
    }
}
