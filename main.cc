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
  std::set<int> keep_t0;
  int keep_prec;

  Cache() {
  //Singlet< Matrix<4, ComplexD > >* gamma = new Singlet< Matrix<4, ComplexD > >[6];
  //for (int i=0;i<6;i++)
  //  initSpinor(gamma[i](),i);

  //std::cout << "Available momenta" << std::endl;
  //for (auto& t : ca.moms) {
  //  std::cout << t.first << std::endl;
  //}

  }

  void fill_peramb(int n,int np,int s,int sp,int t0,int prec,std::vector<ComplexD>& res) {
    //assert(res.size() == NT);
    //printf("%d %d %d %d %d %d\n",n,np,s,sp,t0,prec);
    //for (int t=0;t<NT;t++) {
      //peramb(prec)(t,t0)(n,np)(s,sp) = res[t];
    //}
  }

  void fill_gamma(int n,int np,int s,int sp,int t0,int mu,int prec,std::vector<ComplexD>& res) {
    //assert(res.size() == NT);
    if (mu<3) {
      //for (int t=0;t<NT;t++) {
	//gmu(prec)(mu)(t,t0)(n,np)(s,sp) = res[t];
      //}
    }
  }

  void fill_mom(int mx,int my,int mz,int n,int np,std::vector<ComplexD>& res) {
    //assert(res.size() == NT);
    char buf[64];
    sprintf(buf,"%d_%d_%d",mx,my,mz);
    //auto& m= moms[buf];
    //for (int t=0;t<NT;t++)
    //  m(t)(n,np) = res[t];
  }

  bool operator()(char* tag, std::vector<ComplexD>& res) {
    return false;

    tag+=7;
    if (!strncmp(tag,"pera",4)) {
      tag+=11;
      int prec = *tag - '0';
      int n,np,s,sp,t0;
      tag+=2;
      if (sscanf(tag,"n_%d_%d_s_%d_%d_t_%d",&n,&np,&s,&sp,&t0)==5) {
	fill_peramb(n,np,s,sp,t0,prec,res);
      }
    } else if (!strncmp(tag,"mom",3)) {
      tag+=4;
      int mx,my,mz,n,np;
      if (sscanf(tag,"%d_%d_%d_n_%d_%d",&mx,&my,&mz,&n,&np)==5) {
	fill_mom(mx,my,mz,n,np,res);
      }
    } else if (tag[0]=='G') {
      tag+=1;
      int mu = *tag - '0';
      tag+=6;
      int prec = *tag - '0';
      tag+=2;
      int n,np,s,sp,t0;
      if (sscanf(tag,"n_%d_%d_s_%d_%d_t_%d",&n,&np,&s,&sp,&t0)==5) {
	fill_gamma(n,np,s,sp,t0,mu,prec,res);
      }
    }
    //exit(0);
  }

};

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
void parse(std::string contr, Params& p, Cache<N>& ca, int iter, bool learn) {

  char line[2048];

  if (learn) {
    std::cout << "Parsing iteration " << iter << " of " << contr << std::endl;
  }

  double t0 = dclock();

  FILE* f = fopen(contr.c_str(),"rt");
  assert(f);

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
    } else if (!args[0].compare("GAMMA")) {
    } else if (!args[0].compare("LIGHT_LGAMMA_LIGHT")) {
    } else {
      std::cout << "Unknown command " << args[0] << " in line " << line << std::endl;
      assert(0);
    }

  }

  fclose(f);

  double t1=dclock();

  if (!learn)
    std::cout << "Processing iteration " << iter << " of " << contr << " in " << (t1-t0) << " s" << std::endl;

}

template<int N>
void run(Params& p) {
  Correlators c;
  Cache<N> ca;

  std::vector<std::string> contractions;
  std::vector<std::string> input;
  PADD(p,contractions);
  PADD(p,input);

  int Niter;
  PADD(p,Niter);

  // Define output correlator as well

  // parse what we need
  for (int iter=0;iter<Niter;iter++) {
    for (auto& cc : contractions) {
      parse(cc,p,ca,iter,true);
    }
  }

  // now load inputs with mask for needed parameters
  for (auto& i : input) {
    c.load(i,ca);
  }

  // Compute
  for (int iter=0;iter<Niter;iter++) {
    for (auto& cc : contractions) {
      parse(cc,p,ca,iter,false);
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
