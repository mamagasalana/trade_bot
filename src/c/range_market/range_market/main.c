// bandext.c
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>

static inline int any_true_first_last(const double *lo, const double *hi, npy_intp n,
                                      double val, npy_intp *first, npy_intp *last)
{
    npy_intp i;
    *first = -1; *last = -1;
    for (i=0; i<n; ++i) {
        if (lo[i] <= val && hi[i] >= val) {
            if (*first < 0) *first = i;
            *last = i;
        }
    }
    return (*first >= 0);
}

static PyObject* get_band(PyObject* self, PyObject* args) {
    PyObject *lows_obj, *highs_obj, *prices_obj;
    double threshold, tolerance;
    int VERBOSE;
    if (!PyArg_ParseTuple(args, "OOOddi", &lows_obj, &highs_obj, &prices_obj, &threshold, &tolerance, &VERBOSE))
        return NULL;

        
    PyArrayObject *lows = (PyArrayObject*)PyArray_FROM_OTF(lows_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *highs= (PyArrayObject*)PyArray_FROM_OTF(highs_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *prices=(PyArrayObject*)PyArray_FROM_OTF(prices_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

    if (!lows || !highs || !prices) {
        Py_XDECREF(lows); Py_XDECREF(highs); Py_XDECREF(prices);
        return NULL;
    }

    if (PyArray_SIZE(lows) != PyArray_SIZE(highs)) {
        Py_DECREF(lows); Py_DECREF(highs); Py_DECREF(prices);
        PyErr_SetString(PyExc_ValueError, "lows and highs must have same length");
        return NULL;
    }

    const double *lo = (const double*)PyArray_DATA(lows);
    const double *hi = (const double*)PyArray_DATA(highs);
    const double *pz = (const double*)PyArray_DATA(prices);
    const npy_intp n = PyArray_SIZE(lows);
    const npy_intp m = PyArray_SIZE(prices);

    double best_width = -1.0;
    double best_a = NAN, best_b = NAN;
    double best_emptiness = 0.0;
    int have_best = 0;

    // For returning coverage of the winning band
    PyArrayObject *upper_arr_best = NULL, *lower_arr_best = NULL;
    if (VERBOSE) fprintf(stdout, "%s C compute %s\n", "##############################", "##############################");

    // Main double loop over (a,b) from prices (already unique & sorted on Python side)
    for (npy_intp i=0; i<m; ++i) {
        double a = pz[i];
        for (npy_intp j=i+1; j<m; ++j) {
            double b = pz[j];
            if (b <= a) continue;
            double brange = b - a;

            // -------- pre-check: replicate your "count >= 4" logic quickly
            int count=0, is_support=0, is_resistance=0;
            for (npy_intp k=0; k<n; ++k) {
                int sidx = (lo[k] <= a && hi[k] >= a);
                int ridx = (lo[k] <= b && hi[k] >= b);

                if (sidx && ridx) count += 2;
                if (!is_resistance && ridx) { is_resistance=1; is_support=0; count += 1; }
                if (!is_support    && sidx) { is_support=1;    is_resistance=0; count += 1; }

                if (count >= 4) break;
            }
            if (count < 4) {
                // emptiness would be 1.0; but it can’t beat any candidate (we also require |emptiness-th|<=tol)
                continue;
            }

            // -------- fill-between-trues: find first/last hits for a and b
            npy_intp s_first, s_last, r_first, r_last;
            if (!any_true_first_last(lo, hi, n, a, &s_first, &s_last)) continue;
            if (!any_true_first_last(lo, hi, n, b, &r_first, &r_last)) continue;

            // -------- compute mean coverage for this (a,b)
            double sum_cov = 0.0;
            for (npy_intp k=0; k<n; ++k) {
                int s2 = (s_first>=0 && k>=s_first && k<=s_last);
                int r2 = (r_first>=0 && k>=r_first && k<=r_last);

                if (s2 && r2) {
                    sum_cov += 1.0;
                } else if (r2) {
                    double v = b - lo[k];
                    if (v > 0.0) sum_cov += (v / brange);
                } else if (s2) {
                    double v = hi[k] - a;
                    if (v > 0.0) sum_cov += (v / brange);
                }
                // else add 0
            }
            double mean_cov = sum_cov / (double)n;
            double emptiness = 1.0 - mean_cov;

            if (VERBOSE) fprintf(stdout, "%.5f, %.5f, %.5f\n", emptiness, a, b);
            if (fabs(emptiness - threshold) <= tolerance + 1e-12) { // added epsilon
                double width = brange;
                if (width > best_width) {
                    // (Re)build coverage arrays for the best band
                    npy_intp dims[1] = { n };
                    Py_XDECREF(upper_arr_best);
                    Py_XDECREF(lower_arr_best);
                    upper_arr_best = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
                    lower_arr_best = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
                    if (!upper_arr_best || !lower_arr_best) {
                        Py_XDECREF(lows); Py_XDECREF(highs); Py_XDECREF(prices);
                        Py_XDECREF(upper_arr_best); Py_XDECREF(lower_arr_best);
                        return NULL;
                    }

                    double *upper = (double*)PyArray_DATA(upper_arr_best);
                    double *lower = (double*)PyArray_DATA(lower_arr_best);
                    for (npy_intp k=0; k<n; ++k) { upper[k] = NAN; lower[k] = NAN; }

                    for (npy_intp k=0; k<n; ++k) {
                        int s2k = (s_first>=0 && k>=s_first && k<=s_last);
                        int r2k = (r_first>=0 && k>=r_first && k<=r_last);

                        if (s2k && r2k) {
                            upper[k] = b; lower[k] = a;
                        } else if (r2k) {
                            upper[k] = b;
                            lower[k] = (lo[k] < b ? lo[k] : b); // min(b, low)
                        } else if (s2k) {
                            upper[k] = (hi[k] > a ? hi[k] : a); // max(a, high)
                            lower[k] = a;
                        }
                    }

                    best_width = width;
                    best_a = a; best_b = b;
                    best_emptiness = emptiness;
                    have_best = 1;
                }
            }
        }
    }

    Py_DECREF(lows); Py_DECREF(highs); Py_DECREF(prices);

    if (!have_best) {
        Py_INCREF(Py_None);
        Py_INCREF(Py_None);
        return Py_BuildValue("(O,(d,d),O)", Py_None, NAN, NAN, Py_None);
    } else {
        PyObject *band = Py_BuildValue("(d,d)", best_a, best_b);
        PyObject *cov  = Py_BuildValue("(NN)", upper_arr_best, lower_arr_best); // Steals refs (N)
        PyObject *out  = Py_BuildValue("(d,O,O)", best_emptiness, band, cov);
        Py_DECREF(band);
        Py_DECREF(cov);
        return out;
    }
}

static PyMethodDef Methods[] = {

    {"get_band", get_band, METH_VARARGS,  "get band(\n"},
    { NULL, NULL, 0, NULL }
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "range_market._core",                    /* m_name */
    "Range market in c",  /* m_doc */
    -1,                                  /* m_size */
    Methods                             /* m_methods */
};

PyMODINIT_FUNC PyInit__core(void) {
    import_array();                     /* Initialize NumPy C-API */
    return PyModule_Create(&moduledef);
}



