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
#pragma omp for
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
