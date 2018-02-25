#include <stdio.h>
#include <algorithm>
#include <iostream>
#include <complex>
#include <stdint.h>
#include <assert.h>
#include <map>
#include <vector>
#include <sys/time.h>
#include <memory.h>
#include <stdlib.h>
#include <string.h>
#include <set>
#include "Params.h"
#include <omp.h>

typedef std::complex<double> ComplexD;

inline double dclock() {
  struct timeval tv;
  gettimeofday(&tv,NULL);
  return (1.0*tv.tv_usec + 1.0e6*tv.tv_sec) / 1.0e6;
}

#include "Correlators.h"
#include "Tensor.h"

template<int N>
class Cache {
public:

  std::map<std::string,int> intVals;
  std::map<std::string,std::vector<int> > vintVals;
  std::set<int> keep_t0;
  std::set<std::string> keep_mom;
  int keep_prec;

  std::map< std::string, std::vector< Matrix< N, ComplexD > > > moms;

  Cache() {
  //Singlet< Matrix<4, ComplexD > >* gamma = new Singlet< Matrix<4, ComplexD > >[6];
  //for (int i=0;i<6;i++)
  //  initSpinor(gamma[i](),i);

  //std::cout << "Available momenta" << std::endl;
  //for (auto& t : ca.moms) {
  //  std::cout << t.first << std::endl;
  //}

  }

  bool fill_peramb(int n,int np,int s,int sp,int t0,int prec,std::vector<ComplexD>& res) {

    if (!keep_t0.count(t0))
      return false;

    //assert(res.size() == NT);
    //printf("%d %d %d %d %d %d\n",n,np,s,sp,t0,prec);
    //for (int t=0;t<NT;t++) {
      //peramb(prec)(t,t0)(n,np)(s,sp) = res[t];
    //}

    return true;
  }

  bool fill_gamma(int n,int np,int s,int sp,int t0,int mu,int prec,std::vector<ComplexD>& res) {

    if (!keep_t0.count(t0))
      return false;

    //assert(res.size() == NT);
    if (mu<3) {
      //for (int t=0;t<NT;t++) {
	//gmu(prec)(mu)(t,t0)(n,np)(s,sp) = res[t];
      //}
    }

    return true;
  }

  bool fill_mom(int mx,int my,int mz,int n,int np,std::vector<ComplexD>& res) {
    //assert(res.size() == NT);
    char buf[64];
    sprintf(buf,"%d_%d_%d",mx,my,mz);

    if (!keep_mom.count(buf))
      return false;

    auto& m=moms[buf];
    if (m.size() == 0) {
      m.resize(res.size());
      for (int t=0;t<(int)res.size();t++)
	for (int i=0;i<N;i++)
	  for (int j=0;j<N;j++)
	    m[t](i,j)=NAN;
    }
    for (int t=0;t<(int)res.size();t++)
      m[t](n,np) = res[t];
    return true;
  }

  bool operator()(char* tag, std::vector<ComplexD>& res) {

    tag+=7;
    if (!strncmp(tag,"pera",4)) {
      tag+=11;
      int prec = *tag - '0';
      int n,np,s,sp,t0;
      tag+=2;
      assert(sscanf(tag,"n_%d_%d_s_%d_%d_t_%d",&n,&np,&s,&sp,&t0)==5);
      return fill_peramb(n,np,s,sp,t0,prec,res);
    } else if (!strncmp(tag,"mom",3)) {
      tag+=4;
      int mx,my,mz,n,np;
      assert(sscanf(tag,"%d_%d_%d_n_%d_%d",&mx,&my,&mz,&n,&np)==5);
      return fill_mom(mx,my,mz,n,np,res);
    } else if (tag[0]=='G') {
      tag+=1;
      int mu = *tag - '0';
      tag+=6;
      int prec = *tag - '0';
      tag+=2;
      int n,np,s,sp,t0;
      assert(sscanf(tag,"n_%d_%d_s_%d_%d_t_%d",&n,&np,&s,&sp,&t0)==5);
      return fill_gamma(n,np,s,sp,t0,mu,prec,res);
    }

    return false;

  }

};

template<int N>
std::vector<int> getMomParam(Params& p, Cache<N>& ca, std::vector<std::string>& args, int iarg, int iter) {
  if (args.size() <= iarg) {
    std::cout << "Missing argument " << iarg << " for command " << args[0] << std::endl;
    assert(0);
  }

  char suf[32];
  sprintf(suf,"[%d]",iter);
  auto n = args[iarg] + suf;

  auto f = ca.vintVals.find(n);
  if (f == ca.vintVals.end()) {
    std::vector<int> v;
    std::string sv = p.get(n.c_str());
    std::cout << p.loghead() << "Set " << n << " to " << sv << std::endl;
    p.parse(v,sv);
    ca.vintVals[n] = v;

    assert(v.size() == 3);
    sprintf(suf,"%d_%d_%d",v[0],v[1],v[2]);
    ca.keep_mom.insert(suf);
    return v;
  }  
    
  return f->second;
}

template<int N>
int getIntParam(Params& p, Cache<N>& ca, std::vector<std::string>& args, int iarg, int iter) {
  if (args.size() <= iarg) {
    std::cout << "Missing argument " << iarg << " for command " << args[0] << std::endl;
    assert(0);
  }

  char suf[32];
  sprintf(suf,"[%d]",iter);
  auto n = args[iarg] + suf;

  auto f = ca.intVals.find(n);
  if (f == ca.intVals.end()) {
    int v;
    p.get(n.c_str(),v);
    ca.intVals[n] = v;
    return v;
  }  
    
  return f->second;
}

template<int N>
int getTimeParam(Params& p, Cache<N>& ca, std::vector<std::string>& args, int iarg, int iter, bool isT0) {
  if (args.size() <= iarg) {
    std::cout << "Missing argument " << iarg << " for command " << args[0] << std::endl;
    assert(0);
  }

  char suf[32];
  sprintf(suf,"[%d]",iter);
  auto n = args[iarg] + suf;

  auto f = ca.intVals.find(n);
  if (f == ca.intVals.end()) {
    int v;
    p.get(n.c_str(),v);
    ca.intVals[n] = v;
    if (isT0)
      ca.keep_t0.insert(v);
    return v;
  }  
    
  return f->second;
}

template<int N>
void parse(ComplexD& result, std::string contr, Params& p, Cache<N>& ca, int iter, bool learn) {

  char line[2048];

  if (learn) {
    std::cout << "Parsing iteration " << iter << " of " << contr << std::endl;
  }

  double t0 = dclock();

  FILE* f = fopen(contr.c_str(),"rt");
  assert(f);

  //
  // Logic:
  //
  // Variables: 
  //  Result (complex number)
  //  FACTOR starts a new factor, if previous one was present, add it to result
  //  Current matrix in spin-mode-space; when begintrace set this to unity, when endtrace take trace of it and multiply to current factor

  std::vector<ComplexD> factor;

  //M=M*M; takes about 0.2s this is large?   N*4=240;  240^3 = 13824000 = 0.013 G with O(20 flop/site) -> 1 Gflop/s
  // Now this is without threading, so we can hope for 64 Gflops/s;  still seems slow... ahh... does not use AVX512
  // with avx512 only 25%-30% speedup

  while (!feof(f)) {
    if (!fgets(line,sizeof(line),f))
      break;

    for (int i=strlen(line)-1;i>=0;i--)
      if (line[i]=='\n' || line[i]=='\r' || line[i]==' ')
	line[i]='\0';
      else
	break;

    if (line[0]=='#' || line[0]=='\0')
      continue;

    auto args = split(std::string(line),' ');

    if (!args[0].compare("LIGHT")) {
      int t0 = getTimeParam(p,ca,args,1,iter,false);
      int t1 = getTimeParam(p,ca,args,2,iter,true);

      if (!learn) {
      }
    } else if (!args[0].compare("LIGHTBAR")) {
      int t0 = getTimeParam(p,ca,args,1,iter,true);
      int t1 = getTimeParam(p,ca,args,2,iter,false);

      if (!learn) {
      }
    } else if (!args[0].compare("FACTOR")) {
    } else if (!args[0].compare("BEGINTRACE")) {
    } else if (!args[0].compare("ENDTRACE")) {
    } else if (!args[0].compare("MOM")) {
      std::vector<int> mom = getMomParam(p,ca,args,1,iter);
    } else if (!args[0].compare("GAMMA")) {
      int mu = getIntParam(p,ca,args,1,iter);
      if (!learn) {
      }
    } else if (!args[0].compare("LIGHT_LGAMMA_LIGHT")) {
      int mu = getIntParam(p,ca,args,1,iter);
      if (!learn) {
      }
    } else {
      std::cout << "Unknown command " << args[0] << " in line " << line << std::endl;
      assert(0);
    }

  }

  fclose(f);

  double t1=dclock();

  if (!learn) {
    std::cout << "Processing iteration " << iter << " of " << contr << " in " << (t1-t0) << " s" << std::endl;
    result = 0.0;
    for (auto f : factor) {
      result += f;
    }
  } else {
    assert(factor.size() == 0);
  }

}

template<int N>
void run(Params& p) {
  Correlators c;
  Cache<N> ca;

#if 1
  // TODO: the following code is optimal and hits the memory bandwidth of MCDRAM
  // Make it as stand-alone function and base parse routine around this
  {
    //Matrix< N, Matrix< 4,ComplexD > > M;
    Matrix< 4*N, ComplexD > M;
    memset(&M._internal[0],0,sizeof(ComplexD)*4*N*4*N);

#define Ns 8
#define Nt 60
    Matrix< Ns*Nt, ComplexD > A,B,C;
    memset(&A._internal[0],0,sizeof(ComplexD)*Ns*Nt*Ns*Nt);
    memset(&B._internal[0],0,sizeof(ComplexD)*Ns*Nt*Ns*Nt);
    memset(&C._internal[0],0,sizeof(ComplexD)*Ns*Nt*Ns*Nt);
    double ta,tb;

    for (int iter=0;iter<10;iter++) {
#pragma omp parallel shared(M,ta,tb)
    {
#pragma omp single
     {
       std::cout << "Threads: " << omp_get_num_threads() << std::endl;
      ta=dclock();
     }
     //M=M*M;
     ComplexD* pA = &A._internal[0];
     ComplexD* pB = &B._internal[0];
     ComplexD* pC = &C._internal[0];
#pragma omp for
     for (int ab=0;ab<Ns*Nt*Ns*Nt;ab++) {
       int j=ab / (Ns*Nt);
       int i=ab % (Ns*Nt);
       for (int l=0;l<Ns*Nt;l++)
	 pA[i + Ns*Nt*j] += pB[l + Ns*Nt*i] * pC[l + Ns*Nt*j];
    }
#pragma omp single
     {
      tb=dclock();
     }
    }
    double flopsComplexAdd = 2;
    double flopsComplexMul = 6;
    double flops = pow(Nt*Ns,3.0)*(flopsComplexMul + flopsComplexAdd);
    double gbs = pow(Nt*Ns,3.0)*sizeof(ComplexD)*4.0 / 1024./1024./1024.;
    std::cout << "Performance of matrix mul: " << (flops/1024./1024./1024./(tb-ta)) << " Gflops/s" 
    " and memory bandwidth is " << gbs/(tb-ta) << " GB/s "
    << std::endl;
    }
  }

#endif
  std::vector<std::string> contractions;
  std::vector<std::string> input;
  PADD(p,contractions);
  PADD(p,input);

  int precision;
  PADD(p,precision);
  ca.keep_prec = precision;

  // Define output correlator as well
  std::string output;
  PADD(p,output);
  CorrelatorsOutput co(output);

  // Set output vector size
  int NT;
  std::vector<int> _dt;
  std::vector<std::string> _tag;
  std::vector<double> _scale;
  PADD(p,NT);
  PADD(p,_dt);
  PADD(p,_tag);
  PADD(p,_scale);
  
  int Niter = _dt.size();

  // parse what we need
  for (int iter=0;iter<Niter;iter++) {
    ComplexD res = 0.0;
    for (auto& cc : contractions) {
      parse(res,cc,p,ca,iter,true);
    }
  }

  // now load inputs with mask for needed parameters
  for (auto& i : input) {
    c.load(i,ca);
  }

  // Compute
  for (auto& cc : contractions) {
    std::map<std::string, std::vector<ComplexD> > res;

    for (int iter=0;iter<Niter;iter++) {

      auto f = res.find(_tag[iter]);
      if (f == res.end()) {
	std::vector<ComplexD> r0(NT);
	for (int i=0;i<NT;i++)
	  r0[i] = NAN;
	res[_tag[iter]] = r0;
      }
      f = res.find(_tag[iter]);

      assert(_dt[iter] >= 0 && _dt[iter] < NT);

      ComplexD r = 0.0;
      parse(r,cc,p,ca,iter,false);
      assert(!isnan(r.real())); // important to make remaining logic sound

      ComplexD& t = f->second[_dt[iter]];
      ComplexD v = _scale[iter] * r;
      if (isnan(t.real()))
	t=v;
      else
	t+=v;
    }

    for (auto& f : res) {
      char buf[4096]; // bad even if this likely is large enough
      sprintf(buf,"%s/%s",cc.c_str(),f.first.c_str());
      co.write_correlator(buf,f.second);
    }
  }
}

int main(int argc, char* argv[]) {
  Params p("params.txt");

  int N;
  PADD(p,N);

  if (N==60) {
    run<60>(p);
  } else if (N==120) {
    run<120>(p);
  } else if (N==150) {
    run<150>(p);
  } else if (N==500) {
    run<500>(p);
  } else {
    std::cout << "Unknown basis size " << N << " needs to be added at compile-time for efficiency" << std::endl;
  }

  return 0;
}
