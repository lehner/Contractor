template<class T>
std::vector<T> fold(const std::vector<T>& in) {
  size_t N=in.size();
  std::vector<T> out(N/2+1);
  for (size_t i=0;i<out.size();i++)
    out[i] = 0.5*(in[i] + in[(N-i)%N]);
  return out;
}

class zero_type {
public:
  operator ComplexD() const { return ComplexD(0,0); }
};

zero_type zero;

template<typename sobj>
class Singlet {
public:
  sobj _internal;

  sobj& operator()() { return _internal; }

  Singlet operator*(const Singlet& o) {
    Singlet ret;
    ret._internal = _internal * o._internal;
    return ret;
  }

  Singlet operator+(const Singlet& o) {
    Singlet ret;
    ret._internal = _internal + o._internal;
    return ret;
  }

  friend std::ostream &operator<<(std::ostream &os, const Singlet &m) { 
    return os << "S[ " << m._internal << " ]";
  }

};

template<int N, typename sobj>
class Vector {
public:
  sobj _internal[N];

  Vector() {
  }

  sobj& operator()(int i) { assert(i<N); return _internal[i]; }
};


template<int N, typename sobj>
class Matrix {
public:
  std::vector<sobj> _internal;
 
  sobj& operator()(int i,int j) { return _internal[i+N*j]; }
  const sobj& operator()(int i,int j) const { return _internal[i+N*j]; }

  Matrix() : _internal(N*N) {
  }

  Matrix(const zero_type& zero) : _internal(N*N) {
    for (int i=0;i<N;i++)
      for (int j=0;j<N;j++)
	(*this)(i,j) = zero;
  }

  sobj trace() {
    sobj r = zero;
    for (int i=0;i<N;i++)
      r += (*this)(i,i);
    return r;    
  }

  sobj norm() {
    sobj r = zero;
    for (int i=0;i<N;i++)
      for (int j=0;j<N;j++)
	r += std::norm((*this)(i,j));
    return r;
  }

  friend std::ostream &operator<<(std::ostream &os, const Matrix &m) { 
    os << "M[ " << N << ", ";
    for (int i=0;i<N;i++) {
      os << "[ ";
      for (int j=0;j<N;j++) {
	os << m(i,j);
	if (j<N-1)
	  os<<", ";
      }
      os << " ]";
      if (i<N-1)
	os<<", ";
    }
    return os << " ]";
  }

  friend Matrix conj(const Matrix& o) {
    Matrix<N,sobj> ret;
    for (int i=0;i<N;i++)
      for (int j=0;j<N;j++)
	ret(i,j) = conj(o(j,i));
    return ret;
  }

  template<typename T>
  Matrix operator*(const Singlet<T>& o) const {
    Matrix<N,sobj> ret;
    for (int i=0;i<N;i++)
      for (int j=0;j<N;j++) {
	ret(i,j) = (*this)(i,j) * o._internal;
      }
    return ret;
  }

  Matrix operator*(const ComplexD& o) const {
    Matrix<N,sobj> ret;
    for (int i=0;i<N;i++)
      for (int j=0;j<N;j++) {
	ret(i,j) = (*this)(i,j) * o;
      }
    return ret;
  }

  template<typename T>
  friend Matrix operator*(const Singlet<T>& a,const Matrix<N,sobj>& b) {
    Matrix<N,sobj> ret;
    for (int i=0;i<N;i++)
      for (int j=0;j<N;j++) {
	ret(i,j) = a._internal * b(i,j);
      }
    return ret;
  }

  friend Matrix operator*(const ComplexD& a,const Matrix<N,sobj>& b) {
    Matrix<N,sobj> ret;
    for (int i=0;i<N;i++)
      for (int j=0;j<N;j++) {
	ret(i,j) = a * b(i,j);
      }
    return ret;
  }

  template<typename vobj2>
  Matrix operator*(const Matrix<N,vobj2>& o) const {
    Matrix<N,sobj> ret;
    for (int i=0;i<N;i++)
      for (int j=0;j<N;j++) {
	sobj r = zero;
	for (int l=0;l<N;l++)
	  r+=(*this)(i,l)*o(l,j);
	ret(i,j) = r;
      }
    return ret;
  }

  Matrix operator+(const Matrix& o) const {
    Matrix<N,sobj> ret;
    for (int i=0;i<N;i++)
      for (int j=0;j<N;j++) {
	ret(i,j) = (*this)(i,j) + o(i,j);
      }
    return ret;
  }

  Matrix& operator+=(const Matrix& o) {
    for (int i=0;i<N;i++)
      for (int j=0;j<N;j++) {
	(*this)(i,j) += o(i,j);
      }
    return *this;
  }

  Matrix& operator-=(const Matrix& o) {
    for (int i=0;i<N;i++)
      for (int j=0;j<N;j++) {
	(*this)(i,j) -= o(i,j);
      }
    return *this;
  }

};

/*
gx=np.matrix([ [0,0,0,I], [0,0,I,0], [0,-I,0,0],[-I,0,0,0] ])
gy=np.matrix([ [0,0,0,-1], [0,0,1,0], [0,1,0,0], [-1,0,0,0] ])
gz=np.matrix([ [0,0,I,0], [0,0,0,-I], [-I,0,0,0], [0,I,0,0] ])
gt=np.matrix([ [0,0,1,0], [0,0,0,1], [1,0,0,0], [0,1,0,0] ])
g5=gx*gy*gz*gt
*/

template<typename T>
void initSpinor(Matrix<4, T>& m, int l) {
  int i,j;
  for (i=0;i<4;i++)
    for (j=0;j<4;j++)
      m(i,j)=0.0;
  
  T I(0.0,1.0);
  switch (l) {
  case 0:
    m(0,3)=m(1,2)=I;
    m(2,1)=m(3,0)=-I;
    break;
  case 1:
    m(0,3)=-1.;
    m(1,2)=1.;
    m(2,1)=1.;
    m(3,0)=-1.;
    break;
  case 2:
    m(0,2)=I;
    m(1,3)=-I;
    m(2,0)=-I;
    m(3,1)=I;
    break;
  case 3:
    m(0,2)=1.;
    m(1,3)=1.;
    m(2,0)=1.;
    m(3,1)=1.;
    break;
  case 5:
    m(0,0)=m(1,1)=1.;
    m(2,2)=m(3,3)=-1.;
    break;
  }
}

/*
  Right-multiply spinor

  A = B*spin
  A[i,j] = B[i,l]*spin[l,j]
  
  spin[l,j] is only nonzero for a single value l=spinIdx[j]

  A[i,j] = B[i,spinIdx[j]] * spinCoef[j]
*/

static int spinIdx_0[4] = { 3, 2, 1, 0 };
static ComplexD spinVal_0[4] = { ComplexD(0.0,-1.0), ComplexD(0.0,-1.0), ComplexD(0.0,1.0), ComplexD(0.0,1.0) };

static int spinIdx_1[4] = { 3, 2, 1, 0 };
static ComplexD spinVal_1[4] = { -1.0, 1.0, 1.0, -1.0 };

static int spinIdx_2[4] = { 2, 3, 0, 1 };
static ComplexD spinVal_2[4] = { ComplexD(0.0,-1.0), ComplexD(0.0,1.0), ComplexD(0.0,1.0), ComplexD(0.0,-1.0) };

static int spinIdx_3[4] = { 2, 3, 0, 1 };
static ComplexD spinVal_3[4] = { 1.0, 1.0, 1.0, 1.0 };

static int spinIdx_5[4] = { 0, 1, 2, 3 };
static ComplexD spinVal_5[4] = { 1.0, 1.0, -1.0, -1.0 };

template<int N>
void fast_spin(Matrix<N, ComplexD>& A, Matrix<N, ComplexD>& B, int mu) {
  ComplexD* pA = &A._internal[0];
  ComplexD* pB = &B._internal[0];

  int* spinIdx;
  ComplexD* spinVal;

  switch (mu) {
  case 0:
    spinIdx = spinIdx_0;
    spinVal = spinVal_0;
    break;
  case 1:
    spinIdx = spinIdx_1;
    spinVal = spinVal_1;
    break;
  case 2:
    spinIdx = spinIdx_2;
    spinVal = spinVal_2;
    break;
  case 3:
    spinIdx = spinIdx_3;
    spinVal = spinVal_3;
    break;
  case 5:
    spinIdx = spinIdx_5;
    spinVal = spinVal_5;
    break;
  default:
    std::cout << "Unknown mu=" << mu << std::endl;
    assert(0);
  }
#pragma omp for
  for (int ab=0;ab<N*N;ab++) {
    int j=ab / N;
    int i=ab % N;

    int jn = j / 4;
    int js = j % 4;
    pA[i + N*j] = pB[i + N*(jn*4 + spinIdx[js])] * spinVal[js];
  }
}

/*
  For testing
*/
double randf() {
  return (double)rand() / (double)RAND_MAX - 0.5;
}

ComplexD randc() {
  return ComplexD(randf(),randf());
}

template<int N>
void random_mat(Matrix<N, ComplexD>& A) {
  for (int i=0;i<N;i++)
    for (int j=0;j<N;j++)
      A(i,j)=randc();
}

template<int N>
void identity_mat(Matrix<N, ComplexD>& A) {
  for (int i=0;i<N;i++)
    for (int j=0;j<N;j++)
      A(i,j)=(i==j) ? 1.0 : 0.0;
}


/*
  Fast matrix routines
*/
template<int N>
void fast_zero(Matrix<N, ComplexD>& A) {
  memset(&A._internal[0],0,sizeof(ComplexD)*N);
}

template<int N>
void fast_trans(Matrix<N, ComplexD>& A, Matrix<N, ComplexD>& B) {
  ComplexD* pA = &A._internal[0];
  ComplexD* pB = &B._internal[0];
#pragma omp for
  for (int ab=0;ab<N*N;ab++) {
    int j=ab / N;
    int i=ab % N;
    pA[i + N*j] = pB[j + N*i];
  }
}

template<int N>
void fast_dag(Matrix<N, ComplexD>& A, Matrix<N, ComplexD>& B) {
  ComplexD* pA = &A._internal[0];
  ComplexD* pB = &B._internal[0];
#pragma omp for
  for (int ab=0;ab<N*N;ab++) {
    int j=ab / N;
    int i=ab % N;
    pA[i + N*j] = conj(pB[j + N*i]);
  }
}


template<int N>
void fast_cp(Matrix<N, ComplexD>& A, Matrix<N, ComplexD>& B) {
  ComplexD* pA = &A._internal[0];
  ComplexD* pB = &B._internal[0];
  memcpy(pA,pB,sizeof(ComplexD)*N*N);
}

template<int N>
void fast_mult(Matrix<N, ComplexD>& A, Matrix<N, ComplexD>& B, Matrix<N, ComplexD>& C, Matrix<N, ComplexD>& BT) {
  ComplexD* pA = &A._internal[0];
  fast_trans(BT,B);
  ComplexD* pBT = &BT._internal[0];
  ComplexD* pC = &C._internal[0];
#pragma omp for
  for (int ab=0;ab<N*N;ab++) {
    int j=ab / N;
    int i=ab % N;
    ComplexD r = 0.0;
    for (int l=0;l<N;l++)
      r += pBT[l + N*i] * pC[l + N*j];
    pA[i + N*j] = r;
  }
}

template<int N, int Nmode>
void fast_mult_mode(Matrix<N, ComplexD>& A, Matrix<N, ComplexD>& B, Matrix<Nmode, ComplexD>& C, Matrix<N, ComplexD>& BT) {

  assert(Nmode*4 == N);

  ComplexD* pA = &A._internal[0];
  fast_trans(BT,B);
  ComplexD* pBT = &BT._internal[0];
  ComplexD* pC = &C._internal[0];
#pragma omp for
  for (int ab=0;ab<N*N;ab++) {
    int j=ab / N;
    int i=ab % N;
    int js = j % 4;
    int jn = j / 4;
    ComplexD r = 0.0;
    for (int ln=0;ln<Nmode;ln++)
      r += pBT[ln*4 + js + N*i] * pC[ln + Nmode*jn];
    pA[i + N*j] = r;
  }
}

template<int N>
  void fast_mult_dmode(Matrix<N, ComplexD>& A, std::vector<ComplexD>& v) {

  assert(v.size()*4 == N);

  ComplexD* pA = &A._internal[0];
#pragma omp for
  for (int ab=0;ab<N*N;ab++) {
    int j=ab / N;
    int i=ab % N;
    int jn = j / 4;
    pA[i + N*j] *= v[jn];
  }
}

template<int Nt>
void testFastMatrix() {

  // The following code is optimal and hits the memory bandwidth of MCDRAM
  {
    Matrix< Nt, ComplexD > A,B,C,Ap,_BT_;
    fast_zero(A);
    fast_zero(Ap);
    fast_zero(B);
    fast_zero(C);

    random_mat(B);
    random_mat(C);

    //identity_mat(B);
    //identity_mat(C);

    double ta,tb;

    for (int iter=0;iter<10;iter++) {
#pragma omp parallel
    {
#pragma omp single
     {
       std::cout << "Threads: " << omp_get_num_threads() << std::endl;
      ta=dclock();
     }

     fast_mult(A,B,C,_BT_);

#pragma omp single
     {
      tb=dclock();
     }
    }
    double flopsComplexAdd = 2;
    double flopsComplexMul = 6;
    double flops = pow(Nt,3.0)*(flopsComplexMul + flopsComplexAdd);
    double gbs = pow(Nt,3.0)*sizeof(ComplexD)*4.0 / 1024./1024./1024.;
    std::cout << "Performance of matrix mul: " << (flops/1024./1024./1024./(tb-ta)) << " Gflops/s" 
    " and memory bandwidth is " << gbs/(tb-ta) << " GB/s "
    << std::endl;

    Ap=B*C;

    A -= Ap;
    printf("Norm diff: %g, Norm A: %g\n",A.norm().real(),Ap.norm().real());

    }
  }

}
