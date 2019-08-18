/*
  Authors: Christoph Lehner
  Date: 2017
*/
#include <zlib.h>

static uint32_t crc32_threaded(unsigned char* data, int64_t len, uint32_t previousCrc32 = 0) {
  
  // crc32 of zlib was incorrect for very large sizes, so do it block-wise
  uint32_t crc = previousCrc32;
  off_t blk = 0;
  off_t step = 1024*1024*1024;
  while (len > step) {
    crc = crc32(crc,&data[blk],step);
    blk += step;
    len -= step;
  }
  
  crc = crc32(crc,&data[blk],len);
  return crc;
  
}

class BufferedIO {
 private:
  FILE* f;
  std::vector<char> data;
  size_t buf_size;
  off_t size, pos, pos_in_data;
  std::string fn;

 public:
   BufferedIO(std::string _fn, size_t _buf_size) : buf_size(_buf_size), fn(_fn) {
    f = fopen(fn.c_str(),"rb");
    assert(f);

    fseeko(f,0,SEEK_END);
    size = ftello(f);
    fseeko(f,0,SEEK_SET);
    pos = 0;

    fill();

    pos_in_data = 0;
  }

  void fill() {
    off_t to_read = buf_size;
    off_t left = size - pos;

    assert(left > 0);

    if (to_read > left)
      to_read = left;

    data.resize(to_read);

    double t0=dclock();
    if (fread(&data[0],to_read,1,f)!=1) {
      fprintf(stderr,"Error: %s %llu %llu %llu\n",
	      fn.c_str(),pos,to_read,left);
      exit(2);
    }
    double t1=dclock();

    pos += to_read;

    std::cout << "Read " << (to_read/1024./1024./1024.) << " GB from " << fn << " at " << (size/1024./1024./1024./(t1-t0)) << " GB/s" 
	      << " to offset " << (pos/1024./1024./1024.) << " GB" << " / " << (size/1024./1024./1024.) << " GB"
	      << std::endl;
  }

  bool eof() {
    //std::cout << "Eof: " << pos << " / " << size << " / " << pos_in_data << " / " << data.size() << std::endl;
    return pos_in_data == data.size() && pos == size;
  }

  void get(void* dst, size_t sz) {
    // get all we can from currently filled data
    off_t left = data.size() - pos_in_data;
    if (left >= sz) {
      if (dst)
	memcpy(dst,&data[pos_in_data],sz);
      pos_in_data += sz;
      return;
    }

    if (dst) {
      memcpy(dst,&data[pos_in_data],left);
      dst = (void*)((char*)dst + left);
    }
    sz -= left;

    // if not enough, switch parity, fill, and try to get rest from new parity
    fill();
    pos_in_data = 0;
    get(dst,sz);
  }

  ~BufferedIO() {
    fclose(f);
  }
};

class Correlators {
 public:
  Correlators() {
  }

  ~Correlators() {
  }

  template<typename O>
    void load(std::string fn, O& o) {

    double t2 = dclock();

    BufferedIO bio(fn,1024.*1024.*1024.); // read 1GB at time

    int ntag;
    uint32_t crc32, crc32_comp;
    int NT;
    char buf[4096];

    size_t nadded = 0;
    size_t nskipped = 0;
    std::vector<ComplexD> res;
    std::vector<ComplexD> res0;
    while (!bio.eof()) {

      bio.get(&ntag,4);
      bio.get(buf,ntag);
      bio.get(&crc32,4);
      bio.get(&NT,4);

      if (o(buf,res0)) {
	res.resize(NT);

	bio.get(&res[0],sizeof(ComplexD)*NT);
	
	crc32_comp = crc32_threaded((unsigned char*)&res[0],sizeof(ComplexD)*NT,0x0);

	assert(crc32 == crc32_comp);
      
	o(buf,res);

	nadded++;
      } else {
	nskipped++;
	bio.get(0,sizeof(ComplexD)*NT); // this just skips ahead
      }
    }
    double t3=dclock();

    std::cout << "Parsed " << nadded << " + " << nskipped << " elements in " << t3-t2 << " seconds" << std::endl;
  }


};


class CorrelatorsOutput {
 public:
  FILE* _f;

  CorrelatorsOutput(std::string fn) {

    if (!mpi_id) {
      _f = fopen(fn.c_str(),"w+b");
      assert(_f);
    } else {
      _f = 0;
    }
    
  }

  ~CorrelatorsOutput() {
    if (_f)
      fclose(_f);
  }

  template<class S>
    void write_correlator(char* fn, std::vector<S>& corr, bool verbose = true) {

    if (!_f)
      return;

    int NT = (int)corr.size();
    int ntag = (int)strlen(fn) + 1;
    unsigned short flags = 0;

    std::vector<ComplexD> c(NT);
    for (int i=0;i<NT;i++) {
      c[i] = corr[i]; // first change to complex double no matter what the input was
    }


    uint32_t crc32 = crc32_threaded((unsigned char*)&c[0],sizeof(ComplexD)*NT,0x0);
	  
    if (verbose)
      std::cout << "Wrote " << fn << std::endl;

    assert(fwrite(&ntag,4,1,_f) +
	   fwrite(fn,ntag,1,_f) +
	   fwrite(&crc32,4,1,_f) +
	   fwrite(&NT,2,1,_f) +
	   fwrite(&flags,2,1,_f) +
	   fwrite(&c[0],sizeof(ComplexD)*NT,1,_f) == 6);

    fflush(_f);
  }


};
