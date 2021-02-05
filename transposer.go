package nlp

import (
	"github.com/james-bowman/sparse"
	"gonum.org/v1/gonum/mat"
)

//TransposeTransformer simply transposes the given matrix. Created as a transformer for easy use within a pipeline
type TransposeTransformer struct {
}

// NewTransposeTransformer constructs a new TransposeTransformer.
func NewTransposeTransformer() *TransposeTransformer {
	return &TransposeTransformer{}
}

// Fit does not alter the transformer, it is included for compatibility
func (t *TransposeTransformer) Fit(matrix mat.Matrix) Transformer {
	return t
}

// Transform transposes the matrix and returns the result
func (t *TransposeTransformer) Transform(matrix mat.Matrix) (mat.Matrix, error) {
	if t, isTypeConv := matrix.(sparse.TypeConverter); isTypeConv {
		matrix = t.ToCSR()
	}
	var transposed sparse.CSR
	transposed.Clone(matrix.T().(*sparse.CSC).ToCSR())
	return &transposed, nil
}

// FitTransform is exactly equivalent to calling Fit() followed by Transform() on the
// same matrix.  The returned matrix is a sparse matrix type.
func (t *TransposeTransformer) FitTransform(matrix mat.Matrix) (mat.Matrix, error) {
	if t, isTypeConv := matrix.(sparse.TypeConverter); isTypeConv {
		matrix = t.ToCSR()
	}
	return t.Fit(matrix).Transform(matrix)
}
