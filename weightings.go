package nlp

import (
	"io"
	"math"

	"github.com/james-bowman/sparse"
	"gonum.org/v1/gonum/mat"
)

// TfidfTransformer takes a raw term document matrix and weights each raw term frequency
// value depending upon how commonly it occurs across all documents within the corpus.
// For example a very commonly occurring word like `the` is likely to occur in all documents
// and so would be weighted down.
// More precisely, TfidfTransformer applies a tf-idf algorithm to the matrix where each
// term frequency is multiplied by the inverse document frequency.  Inverse document
// frequency is calculated as log(n/df) where df is the number of documents in which the
// term occurs and n is the total number of documents within the corpus.  We add 1 to both n
// and df before division to prevent division by zero.
// weightPadding can be used to add a value to weights after calculation to make sure terms with zero idf don't get suppressed entirely
// l2Normalization can be used to l2 normalize the values in the matrix after a Transform() is done, done on either each row or each column
type TfidfTransformer struct {
	transform       *sparse.DIA
	weightPadding   float64
	l2Normalization int
}

//L2 Normalization options for the TF-IDF Transformer
const (
	NoL2Normalization = iota
	RowBasedL2Normalization
	ColBasedL2Normalization
)

// NewTfidfTransformer constructs a new TfidfTransformer.
func NewTfidfTransformer() *TfidfTransformer {
	return &TfidfTransformer{}
}

// GetWeightPadding retrieves the weight padding that is added to weights during Fit()
func (t *TfidfTransformer) GetWeightPadding() float64 {
	return t.weightPadding
}

// SetWeightPadding sets the weight padding that is added to weights during Fit()
func (t *TfidfTransformer) SetWeightPadding(wp float64) {
	t.weightPadding = wp
}

//GetL2Normalization retrieves the type of normalization done during Transform()
func (t *TfidfTransformer) GetL2Normalization() int {
	return t.l2Normalization
}

// SetL2Normalization sets the type of normalization done during Transform()
func (t *TfidfTransformer) SetL2Normalization(ln int) {
	t.l2Normalization = ln
}

// Fit takes a training term document matrix, counts term occurrences across all documents
// and constructs an inverse document frequency transform to apply to matrices in subsequent
// calls to Transform().
func (t *TfidfTransformer) Fit(matrix mat.Matrix) Transformer {
	if t, isTypeConv := matrix.(sparse.TypeConverter); isTypeConv {
		matrix = t.ToCSR()
	}
	m, n := matrix.Dims()

	weights := make([]float64, m)
	var df int
	if csr, ok := matrix.(*sparse.CSR); ok {
		for i := 0; i < m; i++ {
			// weight padding can be used to ensure terms with zero idf don't get suppressed entirely.
			weights[i] = math.Log(float64(1+n)/float64(1+csr.RowNNZ(i))) + t.weightPadding
		}
	} else {
		for i := 0; i < m; i++ {
			df = 0
			for j := 0; j < n; j++ {
				if matrix.At(i, j) != 0 {
					df++
				}
			}
			// weight padding can be used to ensure terms with zero idf don't get suppressed entirely.
			weights[i] = math.Log(float64(1+n)/float64(1+df)) + t.weightPadding
		}
	}

	// build a diagonal matrix from array of term weighting values for subsequent
	// multiplication with term document matrics
	t.transform = sparse.NewDIA(m, m, weights)

	return t
}

// Transform applies the inverse document frequency (IDF) transform by multiplying
// each term frequency by its corresponding IDF value.  This has the effect of weighting
// each term frequency according to how often it appears across the whole document corpus
// so that naturally frequent occurring words are given less weight than uncommon ones.
// The returned matrix is a sparse matrix type.
func (t *TfidfTransformer) Transform(matrix mat.Matrix) (mat.Matrix, error) {
	if t, isTypeConv := matrix.(sparse.TypeConverter); isTypeConv {
		matrix = t.ToCSR()
	}
	var product sparse.CSR

	// simply multiply the matrix by our idf transform (the diagonal matrix of term weights)
	product.Mul(t.transform, matrix)

	//Perform L2 normalization of the matrix if the option is selected
	if t.l2Normalization != NoL2Normalization {

		//Transpose the matrix to normalize based on columns
		if t.l2Normalization == ColBasedL2Normalization {
			product.Clone(product.T().(*sparse.CSC).ToCSR())
		}

		rawProduct := product.RawMatrix()

		//Perform normalization
		for i := 0; i < rawProduct.I; i++ {
			sum := 0.0

			for j := rawProduct.Indptr[i]; j < rawProduct.Indptr[i+1]; j++ {
				sum += rawProduct.Data[j] * rawProduct.Data[j]
			}
			if sum == 0.0 {
				continue
			}
			sum = math.Sqrt(sum)
			for j := rawProduct.Indptr[i]; j < rawProduct.Indptr[i+1]; j++ {
				rawProduct.Data[j] /= sum
			}
		}

		//Transpose the matrix back to original format if Column based normalization
		if t.l2Normalization == ColBasedL2Normalization {
			product.Clone(product.T().(*sparse.CSC).ToCSR())
		}
	}

	return &product, nil
}

// FitTransform is exactly equivalent to calling Fit() followed by Transform() on the
// same matrix.  This is a convenience where separate training data is not being
// used to fit the model i.e. the model is fitted on the fly to the test data.
// The returned matrix is a sparse matrix type.
func (t *TfidfTransformer) FitTransform(matrix mat.Matrix) (mat.Matrix, error) {
	if t, isTypeConv := matrix.(sparse.TypeConverter); isTypeConv {
		matrix = t.ToCSR()
	}
	return t.Fit(matrix).Transform(matrix)
}

// Save binary serialises the model and writes it into w.  This is useful for persisting
// a trained model to disk so that it may be loaded (using the Load() method)in another
// context (e.g. production) for reproducible results.
func (t TfidfTransformer) Save(w io.Writer) error {
	_, err := t.transform.MarshalBinaryTo(w)

	return err
}

// Load binary deserialises the previously serialised model into the receiver.  This is
// useful for loading a previously trained and saved model from another context
// (e.g. offline training) for use within another context (e.g. production) for
// reproducible results.  Load should only be performed with trusted data.
func (t *TfidfTransformer) Load(r io.Reader) error {
	var model sparse.DIA

	if _, err := model.UnmarshalBinaryFrom(r); err != nil {
		return err
	}
	t.transform = &model

	return nil
}
