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
#include <omp.h>
#include <mpi.h>

int mpi_n, mpi_id;

#include "Params.h"
typedef std::complex<double> ComplexD;

void glb_sum(std::vector<ComplexD>& v) {
  std::vector<ComplexD> t(v.size());
  MPI_Allreduce(&v[0], &t[0], 2*v.size(), MPI_DOUBLE, MPI_SUM,
		MPI_COMM_WORLD);
  v=t;
}

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
  std::map<std::string,std::vector<ComplexD> > vcVals;
  std::set<int> keep_t0, keep_t0_lgl;
  std::set<std::string> keep_mom;
  int keep_prec;

  std::map< std::string, std::vector< Matrix< N, ComplexD > > > moms;
  std::map< int, std::vector< Matrix< 4*N, ComplexD > > > peramb;
  std::map< int, std::vector< std::map<int, Matrix< 4*N, ComplexD > > > > lgl;

#define SIDX(n,s) ((n)*4 + (s))

  Cache() {
  }

  bool fill_peramb(int n,int np,int s,int sp,int t0,int prec,std::vector<ComplexD>& res) {

    if (!keep_t0.count(t0))
      return false;

    if (prec != keep_prec)
      return false;

    if (!res.size()) {
      assert(n < N);
      assert(np < N);
      return true;
    }

    auto& p=peramb[t0];
    if (p.size() == 0) {
      p.resize(res.size());

      // Remark: don't use threading here!  OMP overhead dominates!
      for (int t=0;t<(int)res.size();t++)
	for (int i=0;i<4*N;i++)
	  for (int j=0;j<4*N;j++)
	    p[t](i,j)=NAN;

    }

    // Remark: don't use threading here!  OMP overhead dominates!
    for (int t=0;t<(int)res.size();t++) {
      p[t](SIDX(n,s),SIDX(np,sp)) = res[t];
    }

    return true;
  }

  bool fill_gamma(int n,int np,int s,int sp,int t0,int mu,int prec,std::vector<ComplexD>& res) {

    if (!keep_t0_lgl.count(t0))
      return false;

    if (prec != keep_prec)
      return false;

    if (!res.size()) {
      assert(n < N);
      assert(np < N);
      return true;
    }

    auto& p=lgl[t0];
    if (p.size() == 0) {
      p.resize(res.size());

      // Remark: don't use threading here!  OMP overhead dominates!
      for (int t=0;t<(int)res.size();t++) {
	auto& m = p[t][mu];
	for (int i=0;i<4*N;i++)
	  for (int j=0;j<4*N;j++)
	    m(i,j)=NAN;
      }

    }

    // Remark: don't use threading here!  OMP overhead dominates!
    for (int t=0;t<(int)res.size();t++) {
      auto& m = p[t][mu];
      m(SIDX(n,s),SIDX(np,sp)) = res[t];
    }

    return true;
  }

  bool fill_mom(int mx,int my,int mz,int n,int np,std::vector<ComplexD>& res) {
    //assert(res.size() == NT);
    char buf[64];
    sprintf(buf,"%d_%d_%d",mx,my,mz);

    if (!keep_mom.count(buf))
      return false;

    if (!res.size()) {
      assert(n < N);
      assert(np < N);
      return true;
    }

    auto& m=moms[buf];
    if (m.size() == 0) {
      m.resize(res.size());

      // Don't use threading here!  OMP overhead dominates.
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
    if (!mpi_id)
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
std::vector<ComplexD> getVCParam(Params& p, Cache<N>& ca, std::vector<std::string>& args, int iarg, int iter) {
  if (args.size() <= iarg) {
    std::cout << "Missing argument " << iarg << " for command " << args[0] << std::endl;
    assert(0);
  }

  char suf[32];
  sprintf(suf,"[%d]",iter);
  auto n = args[iarg] + suf;

  auto f = ca.vcVals.find(n);
  if (f == ca.vcVals.end()) {
    std::vector<ComplexD> v;
    std::string sv = p.get(n.c_str());
    if (!mpi_id)
      std::cout << p.loghead() << "Set " << n << " to " << sv << std::endl;
    p.parse(v,sv);
    ca.vcVals[n] = v;
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

enum T_FLAG { TF_NONE, TF_IST0_PERAMB, TF_IST0_LGL };
template<int N>
int getTimeParam(Params& p, Cache<N>& ca, std::vector<std::string>& args, int iarg, int iter, T_FLAG tf) {
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
    f = ca.intVals.find(n);
  }

  if (tf == TF_IST0_PERAMB)
    ca.keep_t0.insert(f->second);
  else if (tf == TF_IST0_LGL)
    ca.keep_t0_lgl.insert(f->second);

  return f->second;
}

template<int N>
void parse(ComplexD& result, std::string contr, Params& p, Cache<N>& ca, int iter, bool learn) {

  char line[2048];

  if (learn) {
    if (!mpi_id)
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
  Matrix< 4*N, ComplexD > M, tmp, tmp2, res;

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
      int t = getTimeParam(p,ca,args,1,iter,TF_NONE);
      int t0 = getTimeParam(p,ca,args,2,iter,TF_IST0_PERAMB);

      if (!learn) {
	const auto& p = ca.peramb.find(t0);
	if (p == ca.peramb.end())
	  std::cout << "Did not find " << t0 << std::endl;
	assert(p != ca.peramb.end());
#pragma omp parallel
	{
	  fast_mult(res,M, p->second[t], tmp);
	  fast_cp(M,res);
	}
      }
    } else if (!args[0].compare("LIGHTBAR")) {
      int t0 = getTimeParam(p,ca,args,1,iter,TF_IST0_PERAMB);
      int t = getTimeParam(p,ca,args,2,iter,TF_NONE);
      if (!learn) {
	const auto& p = ca.peramb.find(t0);
	assert(p != ca.peramb.end());
#pragma omp parallel
	{
	  fast_spin(res,M,5);
	  fast_dag(tmp2,p->second[t]);
	  fast_mult(M,res, tmp2, tmp);
	  fast_spin(res,M,5);
	  fast_cp(M,res);
	}
      }
    } else if (!args[0].compare("FACTOR")) {
      if (!learn) {
	assert(args.size() >= 3);
	factor.push_back( ComplexD( atof( args[1].c_str()), atof( args[2].c_str() ) ) );
      }
    } else if (!args[0].compare("BEGINTRACE")) {
      if (!learn)
	identity_mat(M);
    } else if (!args[0].compare("ENDTRACE")) {
      if (!learn) {
	assert(factor.size());
	factor[factor.size()-1] *= M.trace();
      }
    } else if (!args[0].compare("MOM")) {
      std::vector<int> mom = getMomParam(p,ca,args,1,iter);
      int t = getTimeParam(p,ca,args,2,iter,TF_NONE);
      if (!learn) {
	assert(mom.size() == 3);
	char buf[64];
	sprintf(buf,"%d_%d_%d",mom[0],mom[1],mom[2]);
	const auto& m=ca.moms.find(buf);
	assert(m != ca.moms.end());
#pragma omp parallel
	{
	  fast_mult_mode(res,M, m->second[t], tmp);
	  fast_cp(M,res);
	}
      }
    } else if (!args[0].compare("MODEWEIGHT")) {
      std::vector<ComplexD> weights = getVCParam(p,ca,args,1,iter);
      if (!learn) {
#pragma omp parallel
	{
	  fast_mult_dmode(M, weights);
	}
      }
    } else if (!args[0].compare("GAMMA")) {
      int mu = getIntParam(p,ca,args,1,iter);
      if (!learn) {
#pragma omp parallel
	{
	  fast_spin(res,M,mu);
	  fast_cp(M,res);
	}
      }
    } else if (!args[0].compare("LIGHT_LGAMMA_LIGHT")) {
      int t0 = getTimeParam(p,ca,args,1,iter,TF_IST0_LGL);
      int t = getTimeParam(p,ca,args,2,iter,TF_NONE);
      int mu = getIntParam(p,ca,args,3,iter);
      int t1 = getTimeParam(p,ca,args,4,iter,TF_IST0_LGL);
      if (!learn) {
	assert(t0 == t1); // only this works so far
	assert(ca.lgl[t0].size() > t);
#pragma omp parallel
	{
	  fast_mult(res,M, ca.lgl[t0][t][mu], tmp);
	  fast_cp(M,res);
	}
      }
    } else {
      std::cout << "Unknown command " << args[0] << " in line " << line << std::endl;
      assert(0);
    }

  }

  fclose(f);

  double t1=dclock();

  if (!learn) {
    if (!mpi_id)
      std::cout << "Processing iteration " << iter << " of " << contr << " in " << (t1-t0) << " s" << std::endl;
    result = 0.0;
    for (auto f : factor) {
      result += f;
    }
  } else {
    //std::cout << "Parsing iteration " << iter << " of " << contr << " in " << (t1-t0) << " s" << std::endl;
    assert(factor.size() == 0);
  }

}

template<int N>
void run(Params& p,int argc,char* argv[]) {
  Correlators c;
  Cache<N> ca;

  if (argc >= 2 && !strcmp(argv[1],"--performance"))
    testFastMatrix<4*N>();

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
  std::vector< std::vector<int> > _dt;
  std::vector< std::vector<std::string> > _tag;
  std::vector< std::vector<double> > _scale;
  PADD(p,NT);
  PADD(p,_dt);
  PADD(p,_tag);
  PADD(p,_scale);
  
  int Niter = _dt.size();
  assert(_tag.size() == Niter);
  assert(_scale.size() == Niter);

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

      ComplexD r = 0.0;

      if (iter % mpi_n == mpi_id)
	parse(r,cc,p,ca,iter,false);

      assert(!isnan(r.real())); // important to make remaining logic sound

      assert(_dt[iter].size() == _scale[iter].size());
      assert(_dt[iter].size() == _tag[iter].size());

      // do as many cuts as we want
      for (int jj=0;jj<_dt[iter].size();jj++) {

	auto f = res.find(_tag[iter][jj]);
	if (f == res.end()) {
	  std::vector<ComplexD> r0(NT);
	  for (int i=0;i<NT;i++)
	    r0[i] = NAN;
	  res[_tag[iter][jj]] = r0;
	}
	f = res.find(_tag[iter][jj]);

	assert(_dt[iter][jj] >= 0 && _dt[iter][jj] < NT);
	
	ComplexD& t = f->second[_dt[iter][jj]];
	ComplexD v = _scale[iter][jj] * r;
	if (isnan(t.real()))
	  t=v;
	else
	  t+=v;

      }
    }

    for (auto& f : res) {
      char buf[4096]; // bad even if this likely is large enough
      sprintf(buf,"%s/%s",cc.c_str(),f.first.c_str());

      glb_sum(f.second);

      co.write_correlator(buf,f.second);
    }
  }
}

int main(int argc, char* argv[]) {
  Params p("params.txt");

  int N;
  PADD(p,N);

  MPI_Init (&argc, &argv);
  MPI_Comm_size (MPI_COMM_WORLD,&mpi_n);
  MPI_Comm_rank (MPI_COMM_WORLD, &mpi_id);

  if (N==60) {
    run<60>(p,argc,argv);
  } else if (N==120) {
    run<120>(p,argc,argv);
  } else if (N==150) {
    run<150>(p,argc,argv);
  } else if (N==500) {
    run<500>(p,argc,argv);
  } else {
    std::cout << "Unknown basis size " << N << " needs to be added at compile-time for efficiency" << std::endl;
  }

  MPI_Finalize();
  return 0;
}
