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
//#include <mkl.h>
extern "C" {
#include <cblas.h>
};

int mpi_n, mpi_id;
int thread_n;

#include "Params.h"
typedef std::complex<double> ComplexD;

void glb_sum(std::vector<ComplexD>& v) {
  std::vector<ComplexD> t(v.size());
  MPI_Allreduce(&v[0], &t[0], 2*v.size(), MPI_DOUBLE, MPI_SUM,
		MPI_COMM_WORLD);
  v=t;
}

void glb_sum(std::vector<char>& v) {
  std::vector<char> t(v.size());
  assert(v.size() < 4294967296);
  MPI_Allreduce(&v[0], &t[0], v.size(), MPI_CHAR, MPI_SUM,
		MPI_COMM_WORLD);
  v=t;
}

void glb_sum(off_t& s) {
  off_t t;
  assert(sizeof(off_t) == sizeof(unsigned long long));
  MPI_Allreduce(&s, &t, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
  s=t;
}

inline double dclock() {
  struct timeval tv;
  gettimeofday(&tv,NULL);
  return (1.0*tv.tv_usec + 1.0e6*tv.tv_sec) / 1.0e6;
}

#include "Correlators.h"
#include "Tensor.h"

struct debug_t {
  std::string file;
  int iter;
};

template<typename T>
class ValueCache {
public:
  std::map<std::string,T> val;
  ValueCache() {
  }
  T& get(const std::string& n, debug_t& debug) {
    const auto& i = val.find(n);
    if (i==val.cend()) {
      fprintf(stderr,"Error: in %s iter %d missing value for %s\n",
	      debug.file.c_str(),debug.iter,n.c_str());
    }
    assert(i!=val.cend());
    return i->second;
  }
  void put(const std::string& n, const T& v) {
    val[n] = v;
  }
  void clear() {
    val.clear();
  }
};


class LocalCache {
public:
  std::map<std::string,int> intVals;
  //std::map<std::string,std::vector<int> > vintVals;
  std::map<std::string,double> realVals;
  std::map<std::string,std::vector<ComplexD> > vcVals;

  LocalCache() {
  }
};

template<int N>
class Cache {
public:

  std::set<int> keep_t0, keep_t0_lgl;
  std::set<std::string> keep_mom;
  int keep_prec;

  std::map< std::string, std::vector< Matrix< N, ComplexD > > > moms;
  std::map< int, std::vector< Matrix< 4*N, ComplexD > > > peramb;
  std::map< int, std::vector< std::map<int, Matrix< 4*N, ComplexD > > > > lgl;

#define NTBASE 65536
#define NTPAIR(a,b) ( (a)*NTBASE + (b) )
#define NTA(p) ( (p) / NTBASE )
#define NTB(p) ( (p) % NTBASE )

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

  bool fill_gamma(int n,int np,int s,int sp,int t0,int t1,int mu,int prec,std::vector<ComplexD>& res) {

    int P = NTPAIR(t0,t1);
    if (!keep_t0_lgl.count(P))
      return false;

    if (prec != keep_prec)
      return false;

    if (!res.size()) {
      assert(n < N);
      assert(np < N);
      return true;
    }

    auto& p=lgl[P];
    if (p.size() == 0) {
      p.resize(res.size());
    }

    // Remark: don't use threading here!  OMP overhead dominates!
    for (int t=0;t<(int)res.size();t++) {
      auto mf = p[t].find(mu);
      if (mf == p[t].end()) {
	auto& m = p[t][mu];

	for (int i=0;i<4*N;i++)
	  for (int j=0;j<4*N;j++)
	    m(i,j)=NAN;

	mf = p[t].find(mu);
	assert(mf != p[t].end());
      }

      mf->second(SIDX(n,s),SIDX(np,sp)) = res[t];
    }

    return true;
  }

  bool fill_mom(char* tag, int mx,int my,int mz,int n,int np,std::vector<ComplexD>& res) {
    //assert(res.size() == NT);
    char buf[2048];
    sprintf(buf,"%s/%d_%d_%d",tag,mx,my,mz);

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
    } else if (tag[0]=='G') {
      tag+=1;
      char* torig = tag;
      int mu = *tag - '0';
      if (*(tag+1)=='G') {
	tag+=2;
	mu=10*mu + (*tag - '0');
	//printf("%d\n",mu);
      }
      tag+=6;
      int prec = *tag - '0';
      tag+=2;
      int n,np,s,sp,t0,t1;
      if (sscanf(tag,"n_%d_%d_s_%d_%d_t_%d_%d",&n,&np,&s,&sp,&t0,&t1)==6) {
	return fill_gamma(n,np,s,sp,t0,t1,mu,prec,res);
      } else if (sscanf(tag,"n_%d_%d_s_%d_%d_t_%d",&n,&np,&s,&sp,&t0)==5) {
	return fill_gamma(n,np,s,sp,t0,t0,mu,prec,res);
      } else {
	printf("# %s | %s\n",tag,torig); //0_prec1/n_0_0_s_0_0_t_0
	assert(0);
      }
    } else {
      char cat[2048];
      int mx,my,mz,n,np;
      
      assert(strlen(tag) < 2048);
      strcpy(cat,tag);

      char* slash = strrchr(cat,'/');
      if (slash) {
	*slash = '\0';
	slash++;
	
	if (sscanf(slash,"%d_%d_%d_n_%d_%d",&mx,&my,&mz,&n,&np)!=5) {
	  fprintf(stderr,"Tag: %s\n",tag);
	  exit(2);
	}
	return fill_mom(cat,mx,my,mz,n,np,res);
      }
    }

    return false;

  }

};

template<int N>
std::string getMomParam(Params& p, Cache<N>& ca, std::vector<std::string>& args, int iarg, int iter, bool learn) {
  if (args.size() <= iarg) {
    std::cout << "Missing argument " << iarg << " for command " << args[0] << std::endl;
    assert(0);
  }

  char suf[32];

  bool isConst = (args[iarg][0] == '[');

  std::string n;
  if (isConst) {
    n = args[iarg];
  } else {
    sprintf(suf,"[%d]",iter);
    n = args[iarg] + suf;
  }

  //auto f = ca.vintVals.find(n);
  //if (f == ca.vintVals.end()) 
  {
    std::vector<std::string> v;
    std::string sv = isConst ? n.substr(1,n.length()-2) : p.get(n.c_str());
    if (!mpi_id)
      std::cout << p.loghead() << "Set " << n << " to " << sv << std::endl;
    p.parse(v,sv);
    //ca.vintVals[n] = v;
      
    std::string buf;
    if (v.size() == 3) {
      buf="mom/" + v[0] + "_" + v[1] + "_" + v[2];
    } else if (v.size() == 4) {
      buf=v[3] + "/" + v[0] + "_" + v[1] + "_" + v[2];
    } else {
      fprintf(stderr,"Unknown matrix %s\n",sv.c_str());
      exit(2);
    }

    if (learn)
      ca.keep_mom.insert(buf);
    return buf;
  }  
}

//template<int N>
std::vector<ComplexD> getVCParam(Params& p, LocalCache& lc, std::vector<std::string>& args, int iarg, int iter) {
  if (args.size() <= iarg) {
    std::cout << "Missing argument " << iarg << " for command " << args[0] << std::endl;
    assert(0);
  }

  bool isConst = (args[iarg][0] == '[');

  std::string n;
  char suf[32];
  if (isConst) {
    n = args[iarg];
  } else {
    sprintf(suf,"[%d]",iter);
    n = args[iarg] + suf;
  }

  auto f = lc.vcVals.find(n);
  if (f == lc.vcVals.end()) {
    std::vector<ComplexD> v;
    std::string sv = isConst ? n.substr(1) : p.get(n.c_str());
    if (!mpi_id)
      std::cout << p.loghead() << "Set " << n << " to " << sv << std::endl;
    p.parse(v,sv);
    lc.vcVals[n] = v;
    return v;
  }  
    
  return f->second;
}

//template<int N>
int getIntParam(Params& p, LocalCache& lc, std::vector<std::string>& args, int iarg, int iter) {
  if (args.size() <= iarg) {
    std::cout << "Missing argument " << iarg << " for command " << args[0] << std::endl;
    assert(0);
  }

  bool isConst = (args[iarg][0] >= '0' && args[iarg][0] <= '9');

  char suf[32];
  std::string n;

  if (isConst) {
    n = args[iarg];
  } else {
    sprintf(suf,"[%d]",iter);
    n = args[iarg] + suf;
  }

  auto f = lc.intVals.find(n);
  if (f == lc.intVals.end()) {
    int v;
    if (isConst) {
      v = atoi(n.c_str());
    } else {
      p.get(n.c_str(),v);
    }
    lc.intVals[n] = v;
    return v;
  }  
    
  return f->second;
}

//template<int N>
double getRealParam(Params& p, LocalCache& lc, std::vector<std::string>& args, int iarg, int iter) {
  if (args.size() <= iarg) {
    std::cout << "Missing argument " << iarg << " for command " << args[0] << std::endl;
    assert(0);
  }

  bool isConst = (args[iarg][0] >= '0' && args[iarg][0] <= '9') || (args[iarg][0] == '-');

  char suf[32];
  std::string n;

  if (isConst) {
    n = args[iarg];
  } else {
    sprintf(suf,"[%d]",iter);
    n = args[iarg] + suf;
  }

  auto f = lc.realVals.find(n);
  if (f == lc.realVals.end()) {
    double v;
    if (isConst) {
      v = atof(n.c_str());
    } else {
      p.get(n.c_str(),v);
    }
    lc.realVals[n] = v;
    return v;
  }  
    
  return f->second;
}

enum T_FLAG { TF_NONE, TF_IST0_PERAMB };
template<int N>
int getTimeParam(Params& p, Cache<N>& ca, LocalCache& lc, std::vector<std::string>& args, int iarg, int iter, T_FLAG tf, bool learn) {
  if (args.size() <= iarg) {
    std::cout << "Missing argument " << iarg << " for command " << args[0] << std::endl;
    assert(0);
  }

  char suf[32];
  sprintf(suf,"[%d]",iter);
  auto n = args[iarg] + suf;

  auto f = lc.intVals.find(n);
  if (f == lc.intVals.end()) {
    int v;
    p.get(n.c_str(),v);
    lc.intVals[n] = v;
    f = lc.intVals.find(n);
  }

  if (learn) {
    if (tf == TF_IST0_PERAMB)
      ca.keep_t0.insert(f->second);
  }

  return f->second;
}

class Perf {
public:
  struct _pi_ { double time; int N; _pi_() : time(0), N(0) {} };
  std::map<std::string,_pi_> stat;
  double t0;
  void begin(std::string a) {
    t0=dclock();
  }
  void end(std::string a) {
    auto&s=stat[a]; s.time += dclock()-t0; s.N++;
  }
  void print() {
    for (auto s : stat)
      std::cout << "Performance " << s.first << " took " << s.second.time << " s over "
		<< s.second.N << " iterations at " << s.second.time / s.second.N << " s/iter" << std::endl; 
  }
};

class File {
public:
  std::vector<std::string> lines;
  size_t cur;
  File() {
    cur=0;
  }

  File(const char* fn, const char* fmt) {
    char line[2048];
    FILE* f = fopen(fn,fmt);
    if (!f) {
      fprintf(stderr,"Failed to open %s\n",fn);
      exit(1);
    }
    while (!::feof(f)) {
      if (!::fgets(line,sizeof(line),f))
	break;
      lines.push_back(line);
    }
    fclose(f);
    cur=0;
  }

  bool feof() {
    return cur == lines.size();
  }

  char* fgets(char* str, int num) {
    if (feof())
      return 0;
    strncpy(str,lines[cur++].c_str(),num);
    return str;
  }

  void close() {
    cur=0;
  }

  void reset() {
    cur=0;
  }
};

class FileCache {
public:
  std::map<std::string,File> files;
  File& open(const char* fn, const char* fmt) {
    std::string tag = std::string(fn) + "|" + fmt;
    auto f = files.find(tag);
    if (f==files.end()) {
      files[tag] = File(fn,fmt);
      return files[tag];
    }
    f->second.reset();
    return f->second;
  }
};

template<int N>
void parse(ComplexD& result, std::string contr, Params& p, LocalCache& lc, Cache<N>& ca, int iter, bool learn, ValueCache<ComplexD>& vc,  
	   ValueCache< Matrix< 4*N, ComplexD > >& mc, FileCache& fc) {

  debug_t debug;
  
  debug.file = contr;
  debug.iter = iter;

  char line[2048];

  if (learn) {
    if (!mpi_id)
      std::cout << "Parsing iteration " << iter << " of " << contr << std::endl;
  }

  double t0 = dclock();

  File& f = fc.open(contr.c_str(),"rt");

  //
  // Logic:
  //
  // Variables: 
  //  Result (complex number)
  //  FACTOR starts a new factor, if previous one was present, add it to result
  //  Current matrix in spin-mode-space; when begintrace set this to unity, when endtrace take trace of it and multiply to current factor

  std::vector<ComplexD> factor;
  Matrix< 4*N, ComplexD > M, tmp, tmp2, res;
  Matrix< N, ComplexD > tmpc;
  std::vector<std::string> ttr;
  bool trace_open = false;

  Perf perf;
  while (!f.feof()) {
    if (!f.fgets(line,sizeof(line)))
      break;

    for (int i=strlen(line)-1;i>=0;i--)
      if (line[i]=='\n' || line[i]=='\r' || line[i]==' ')
	line[i]='\0';
      else
	break;

    if (line[0]=='#' || line[0]=='\0')
      continue;

    auto args = split(std::string(line),' ');

    if (!learn)
      perf.begin(args[0]);

    if (!args[0].compare("LIGHT")) {
      int t = getTimeParam(p,ca,lc,args,1,iter,TF_NONE, learn);
      int t0 = getTimeParam(p,ca,lc,args,2,iter,TF_IST0_PERAMB, learn);

      if (!learn) {
	const auto& p = ca.peramb.find(t0);
	if (p == ca.peramb.end())
	  std::cout << "Did not find " << t0 << std::endl;
	assert(p != ca.peramb.end());
	fast_mult(res,M, p->second[t], tmp);
	fast_cp(M,res);
      }
    } else if (!args[0].compare("LIGHTBAR")) {
      int t0 = getTimeParam(p,ca,lc,args,1,iter,TF_IST0_PERAMB, learn);
      int t = getTimeParam(p,ca,lc,args,2,iter,TF_NONE, learn);
      if (!learn) {
	const auto& p = ca.peramb.find(t0);
	if (p == ca.peramb.end()) {
	  printf("-----\n");
	  printf("t0 = %d\n",t0);
	  printf("iter = %d\n",iter);
	  printf("contr = %s\n",contr.c_str());
	  for (auto& x : ca.peramb) {
	    printf("Has %d\n",x.first);
	  }
	  printf("-----\n");
	  fflush(stdout);
	}
	assert(p != ca.peramb.end());
	//#pragma omp parallel
	{
	  fast_spin(res,M,5);
	  fast_dag(tmp2,p->second[t]);
	}
	fast_mult(M,res, tmp2, tmp);
	//#pragma omp parallel
	{
	  fast_spin(res,M,5);
	}
	fast_cp(M,res);
      }
    } else if (!args[0].compare("FACTOR")) {
      if (!learn) {
	assert(args.size() >= 3);
	factor.push_back( ComplexD( atof( args[1].c_str()), atof( args[2].c_str() ) ) );
      }
    } else if (!args[0].compare("BEGINTRACE")) {
      if (!learn) {
	assert(!trace_open);
	identity_mat(M); // TODO: make identity_mat faster, threading
	trace_open=true;
      }
    } else if (!args[0].compare("BEGINDEFINE")) {
      if (!learn) {
	factor.push_back( 1.0 );
      }
    } else if (!args[0].compare("ENDDEFINE")) {
      assert(args.size() == 2);
      if (!learn) {
	assert(!trace_open);
	assert(factor.size());
	ComplexD val = factor[factor.size()-1];
	vc.put(args[1],val);
	factor.pop_back();
      }
    } else if (!args[0].compare("BEGINMDEFINE")) {
      if (!learn) {
	if (args.size() == 1) {
	  identity_mat(M);
	} else if (args.size() == 3) {
	  identity_mat(M,ComplexD( atof( args[1].c_str()), atof( args[2].c_str()) ));
	} else {
	  fprintf(stderr,"Invalid number of arguments for BEGINMDEFINE!\n");
	  exit(2);
	}
      }
    } else if (!args[0].compare("ENDMDEFINE")) {
      if (!learn) {
	if (args.size() == 2) {
	  mc.put(args[1],M);
	} else if (args.size() == 3) {
	  assert(!args[2].compare("+"));
	  tmp = mc.get(args[1],debug);
	  //#pragma omp parallel
	  {
	    fast_addto(tmp,M);
	  }
	  mc.put(args[1],tmp);
	} else {
	  fprintf(stderr,"Invalid syntax of ENDMDEFINE!\n");
	  exit(3);
	}
      }
    } else if (!args[0].compare("EVAL")) {
      assert(args.size() == 2);
      if (!learn) {
	assert(factor.size());
	factor[factor.size()-1] *= vc.get(args[1],debug);
      }
    } else if (!args[0].compare("EVALM")) {
      assert(args.size() == 2);
      if (!learn) {
	fast_mult(res,M, mc.get(args[1],debug), tmp);
	fast_cp(M,res);
      }
    } else if (!args[0].compare("EVALMDAG")) {
      assert(args.size() == 2);
      if (!learn) {
	//#pragma omp parallel
	{
	  fast_dag(tmp2,mc.get(args[1],debug));	
	}
	fast_mult(res,M, tmp2, tmp); // dag
	fast_cp(M,res);
      }
    } else if (!args[0].compare("ENDTRACE")) {
      if (!learn) {
	assert(trace_open);
	assert(factor.size());
	factor[factor.size()-1] *= M.trace();
	trace_open=false;
      }
    } else if (!args[0].compare("MOM")) {
      std::string buf = getMomParam(p,ca,args,1,iter,learn);
      int t = getTimeParam(p,ca,lc,args,2,iter,TF_NONE, learn);
      
      if (!learn) {
	const auto& m=ca.moms.find(buf);
	if (m == ca.moms.end()) {
	  fprintf(stderr,"Could not find matrix %s\n",buf.c_str());
	  exit(1);
	}
	//#pragma omp parallel
 	{
	  fast_mult_mode(res,M, m->second[t], tmp);
	}
	fast_cp(M,res);
      }
    } else if (!args[0].compare("MOMDAG")) {
      std::string buf = getMomParam(p,ca,args,1,iter,learn);
      int t = getTimeParam(p,ca,lc,args,2,iter,TF_NONE,learn);
      
      if (!learn) {
	const auto& m=ca.moms.find(buf);
	if (m == ca.moms.end()) {
	  fprintf(stderr,"Could not find matrix %s\n",buf.c_str());
	  exit(1);
	}
	//#pragma omp parallel
 	{
	  fast_dag(tmpc,m->second[t]);
	  fast_mult_mode(res,M, tmpc, tmp);
	}
	fast_cp(M,res);
      }
    } else if (!args[0].compare("MODEWEIGHT")) {
      std::vector<ComplexD> weights = getVCParam(p,lc,args,1,iter);
      if (!learn) {
	//#pragma omp parallel
	{
	  fast_mult_dmode(M, weights);
	}
      }
    } else if (!args[0].compare("MODECUT")) {
      double lambda = getRealParam(p,lc,args,1,iter);
      if (!learn) {
	std::vector<ComplexD> weights(N);
	for (int i=0;i<N;i++)
	  weights[i] = ((double)i < (double)N*lambda) ? 1.0 : 0.0;
	//#pragma omp parallel
	{
	  fast_mult_dmode(M, weights);
	}
      }
    } else if (!args[0].compare("GAMMA")) {
      int mu = getIntParam(p,lc,args,1,iter);
      if (!learn) {
	//#pragma omp parallel
	{
	  fast_spin(res,M,mu);
	}
	fast_cp(M,res);
      }
    } else if (!args[0].compare("LIGHT_LGAMMA_LIGHT")) {
      int t0 = getTimeParam(p,ca,lc,args,1,iter,TF_NONE,learn);
      int t = getTimeParam(p,ca,lc,args,2,iter,TF_NONE,learn);
      int mu = getIntParam(p,lc,args,3,iter);
      int t1 = getTimeParam(p,ca,lc,args,4,iter,TF_NONE,learn);
      int p=NTPAIR(t0,t1);
      if (!learn) {
	assert(ca.lgl[p].size() > t);
	fast_mult(res,M, ca.lgl[p][t][mu], tmp);
	fast_cp(M,res);
      } else {
	ca.keep_t0_lgl.insert(p);
      }
    } else {
      std::cout << "Unknown command " << args[0] << " in line " << line << std::endl;
      assert(0);
    }
    if (!learn)
      perf.end(args[0]);
  }

  f.close();

  double t1=dclock();

  if (!learn) {
    if (!mpi_id) {
      std::cout << "Processing iteration " << iter << " of " << contr << " in " << (t1-t0) << " s" << std::endl;
      perf.print();
    }
    result = 0.0;
    for (auto f : factor) {
      result += f;
    }
  } else {
    //std::cout << "Parsing iteration " << iter << " of " << contr << " in " << (t1-t0) << " s" << std::endl;
    assert(factor.size() == 0);
  }

}

bool has(std::vector<std::string>& a, std::string& c) {
  for (auto& x : a)
    if (!x.compare(c))
      return true;
  return false;
}

template<int N>
void run(Params& p,int argc,char* argv[]) {
  Correlators c;

#pragma omp parallel
  {
    if (argc >= 2 && !strcmp(argv[1],"--performance"))
      testFastMatrix<4*N>();
  }

  struct {
    std::vector<std::string> file;
    std::vector<std::string> tag;
  } contraction;

  std::vector<std::string> input;
  PADD(p,contraction.file);
  PADD(p,contraction.tag);

  assert(contraction.file.size() == contraction.tag.size());

  PADD(p,input);
  
  int precision;
  PADD(p,precision);

  Cache<N> ca;
  ca.keep_prec = precision;
  
  // Define output correlator as well
  std::string output;
  PADD(p,output);
  CorrelatorsOutput co(output);
  
  // Set output vector size
  int NT;
  std::vector< std::vector<int> > _dt;
  std::vector< std::vector<std::string> > _tag;
  std::vector< std::vector<std::string> > _contractions;
  std::vector< std::vector<double> > _scale;
  std::vector< std::string > _header;
  PADD(p,NT);
  PADD(p,_dt);
  PADD(p,_tag);
  PADD(p,_contractions);
  PADD(p,_scale);
  PADD(p,_header);

  int Niter = _dt.size();
  assert(_tag.size() == Niter);
  assert(_scale.size() == Niter);
  assert(_contractions.size() == Niter);
  assert(_header.size() == Niter);

  {
    ValueCache<ComplexD> _vc;
    ValueCache< Matrix< 4*N, ComplexD > > _mc;
    FileCache _fc;
    LocalCache _lc;

    // parse what we need
    for (int iter=0;iter<Niter;iter++) {
      ComplexD res = 0.0;
      
      if (iter % mpi_n == mpi_id) {
	if (_header[iter].size())
	  parse(res,_header[iter],p,_lc,ca,iter,true,_vc,_mc,_fc);
	for (int jj=0;jj<(int)contraction.file.size();jj++) {
	  auto& cc = contraction.file[jj];
	  auto& tt = contraction.tag[jj];
	  if (has(_contractions[iter],tt)) {
	    parse(res,cc,p,_lc,ca,iter,true,_vc,_mc,_fc);
	  }
	}
      }
    }

    // now load inputs with mask for needed parameters
    for (auto& i : input) {
      c.load(i,ca);
    }
  }

  std::vector< std::map<std::string, std::vector<ComplexD> > >  thread_res(thread_n);

#pragma omp parallel
  {
    // thread id
    int thread_id = omp_get_thread_num();

    // each thread needs own value cache
    ValueCache<ComplexD> vc;
    ValueCache< Matrix< 4*N, ComplexD > > mc;
    FileCache fc;
    LocalCache lc;

    // Create
    std::map<std::string, std::vector<ComplexD> > & res = thread_res[thread_id];
    
    for (int iter=0;iter<Niter;iter++) {
      
      assert(_dt[iter].size() == _scale[iter].size());
      assert(_dt[iter].size() == _contractions[iter].size());
      assert(_dt[iter].size() == _tag[iter].size());
      
      for (int jj=0;jj<_dt[iter].size();jj++) {
	std::string tt = _tag[iter][jj];
	auto f = res.find(tt);
	if (f == res.end()) {
	  std::vector<ComplexD> r0(NT);
	  for (int i=0;i<NT;i++)
	    r0[i] = NAN;
	  res[tt] = r0;
	}
	
	f = res.find(tt);
	ComplexD& t = f->second[_dt[iter][jj]];	  
	t=0.0;
      }
    }
    
    // Compute

    //int parallel_n = mpi_n * thread_n;
    //int parallel_id = thread_id + thread_n * mpi_id;

    for (int iter=0;iter<Niter;iter++) {
      
      ComplexD r = 0.0;
      std::map<std::string,ComplexD> rm;
      
      if (iter % mpi_n == mpi_id) {

	int local_iter = iter / mpi_n;
	if (local_iter % thread_n != thread_id)
	  continue;
	
	vc.clear();
	mc.clear();

	if (_header[iter].size())
	  parse(r,_header[iter],p,lc,ca,iter,false,vc,mc,fc);

	
	for (int jj=0;jj<(int)contraction.file.size();jj++) {
	  auto& cc = contraction.file[jj];
	  auto& tt = contraction.tag[jj];
	  auto& rmm = rm[tt];
	  
	  if (has(_contractions[iter],tt)) {
	    rmm = 0.0;
	    parse(rmm,cc,p,lc,ca,iter,false,vc,mc,fc);
	  }
	}

	// do as many cuts as we want
	for (int jj=0;jj<_dt[iter].size();jj++) {
	  
	  std::string tt = _tag[iter][jj];
	  auto f = res.find(tt);
	  assert(f != res.end());
	  
	  assert(_dt[iter][jj] >= 0 && _dt[iter][jj] < NT);
	    
	  ComplexD& t = f->second[_dt[iter][jj]];
	  assert(rm.find(_contractions[iter][jj]) != rm.end());
	  ComplexD v = _scale[iter][jj] * rm[_contractions[iter][jj]];
	  t+=v;
	  
	}
      }
    }
  }
  
  for (auto& f : thread_res[0]) {
    char buf[4096]; // bad even if this likely is large enough
    sprintf(buf,"%s",f.first.c_str());

    // first sum over threads
    std::vector<ComplexD> acc(f.second.size(), 0.0);
    for (int i=0;i<thread_n;i++) {
      auto & tres = thread_res[i];
      for (int j=0;j<f.second.size();j++) {
	acc[j] += tres.find(f.first)->second[j];
      }
    }
    
    glb_sum(acc);

    co.write_correlator(buf,acc);
  }
}

int main(int argc, char* argv[]) {
  Params p("params.txt");

  int N;
  PADD(p,N);

  MPI_Init (&argc, &argv);
  MPI_Comm_size (MPI_COMM_WORLD,&mpi_n);
  MPI_Comm_rank (MPI_COMM_WORLD, &mpi_id);

#pragma omp parallel
  {
    thread_n = omp_get_num_threads();
  }
  
  std::cout << "Here MPI rank " << mpi_id << " / " << mpi_n << " has " << thread_n << " threads" << std::endl;
  
  if (N==60) {
    run<60>(p,argc,argv);
  } else if (N==120) {
    run<120>(p,argc,argv);
  } else if (N==150) {
    run<150>(p,argc,argv);
  } else if (N==200) {
    run<500>(p,argc,argv);
  } else {
    std::cout << "Unknown basis size " << N << " needs to be added at compile-time for efficiency" << std::endl;
  }
  
  MPI_Finalize();
  return 0;
}
