/*
 * =====================================================================================
 *
 *       Filename:  mopsr_dbsvddb.c
 *
 *    Description:  Performs SVD based RFI cleaning 
 *
 *        Version:  1.0
 *        Created:  05/02/18 19:31:07
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Wael Farah
 *   Organization:  
 *
 * =====================================================================================
 */
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <getopt.h>
#include <time.h>

#include "mkl.h"

//#include "lapacke.h"
//#include "cblas.h"

#include "dada_client.h"
#include "dada_hdu.h"

#include "ascii_header.h"
#include "daemon.h"


#define MIN_SVD_NSAMPLES 50

#ifndef MAX
#define MAX(a,b) ((a)>(b) ?(a):(b))
#endif
#ifndef MIN
#define MIN(a,b) ((a)<(b) ?(a):(b))
#endif

#ifdef mkl_malloc
#define alignment 64
#define MALLOC(a) (mkl_malloc((a),alignment))
#define CALLOC(a,b) (mkl_calloc((a),(b),alignment))
#define FREE(a) (mkl_free(a))
#else
#define MALLOC(a) (malloc(a))
#define CALLOC(a,b) (calloc(a,b))
#define FREE(a) (free(a))
#endif

// set to use double precission
//#define USEDOUBLE

#ifdef USEDOUBLE
typedef double w_dtype;
#else
typedef float w_dtype;
#endif

// Input/output buffer type, assume 32bit float
typedef float io_dtype;


int get_data(char*, uint64_t, uint64_t, uint64_t, w_dtype**);
int dbsvddb_open(dada_client_t*);
int dbsvddb_close(dada_client_t*, uint64_t);
int64_t dbsvddb_write(dada_client_t*, void*, uint64_t);
int64_t dbsvddb_io_block(dada_client_t*, void*, uint64_t, uint64_t);
int free_work(dada_client_t*);
int malloc_work(dada_client_t*);

int dbsvddb_transpose_SFT_FST(void *, w_dtype**, uint64_t, uint64_t, uint64_t);
int dbsvddb_transpose_STF_FST(void *, w_dtype**, uint64_t, uint64_t, uint64_t);
int dbsvddb_write_FST_SFT(char*, w_dtype*, uint64_t, uint64_t, uint64_t);
int dbsvddb_write_FST_STF(char*, w_dtype*, uint64_t, uint64_t, uint64_t);
static inline int dbsvddb_get_neig(w_dtype*, int, int, float);


void usage()
{
  fprintf(stdout,
      "mopsr_dbsvddb [options] in_key out_key\n"
      " -p [nprocs]  number of processes to spawn [default: 1]\n"
      " -n [neig]    minimum number of eigenvectors to keep in input data [default: 5]\n"
      " -t [thresh]  rejection threshold used for eigenvalue ratios [default: 1.3]\n"
      " -d           daemonize\n"
      " -vV          increase verbosity mode\n"
      " -s           1 transfer, then exit\n"
      " -D           path to eigen-zap file [default: ./out.eig]\n"
      " -m           do not create monitoring statistics\n"
      " -h           show help\n"
      " in_key       DADA key for input data block\n"
      " out_key      DADA key for output data block\n");
}


typedef struct {
  dada_hdu_t*    hdu;
  key_t          key;
  uint64_t       block_size;
  uint64_t       bytes_written;
  unsigned       block_open;
  char*          curr_block;
} mopsr_dbsvddb_hdu_t;


typedef struct {
  w_dtype **U;
  w_dtype **Vt;
  w_dtype **W;
  w_dtype **superb;
  w_dtype **PC;
  w_dtype **data;
  w_dtype *outdata;
  lapack_int *info;
  lapack_int lwork;
} mopsr_dbsvddb_work_t;

typedef struct {
  mopsr_dbsvddb_hdu_t output;

  uint64_t bytes_in;

  uint64_t bytes_out;

  int verbose;
  int nprocs;

  unsigned int nchans;
  unsigned int nbeams;
  unsigned int nsamples;

  int min_neig;
  float thresh;

  unsigned int nbit;


  unsigned quit;
  int nW;
  int * eigs_zapped;
  int write_monitoring;
  FILE * eigs_zapped_file;
  char eigs_zapped_path[1024];
  char order[4];
  mopsr_dbsvddb_work_t work;
} mopsr_dbsvddb_t;

#define DADA_DBNUM_INIT {{0},0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,"0","0",{0}}



int main(int argc, char* argv[])
{
  mopsr_dbsvddb_t dbsvddb = DADA_DBNUM_INIT;

  dada_hdu_t* in_hdu = 0;
  
  dada_client_t* client = 0;

  // input data block HDU key
  key_t in_key = 0;

  unsigned single_transfer = 0;

  char daemon = 0;
  char verbose = 0;

  int min_neig = 5;
  float thresh = 1.3;
  sprintf(dbsvddb.eigs_zapped_path,"out.eig");

  int arg = 0;

  int write_monitoring = 1;
  
  while ((arg=getopt(argc,argv,"p:n:t:mdvVs")) != -1)
  {
    switch (arg)
    {
      case 'p':
        if (sscanf (optarg, "%i", &(dbsvddb.nprocs)) != 1)
        {
          fprintf(stderr,"ERROR: could not parse nprocs from %s\n",optarg);
          usage();
          return EXIT_FAILURE;
        }
        break;

      case 's':
        single_transfer = 1;
        break;

      case 'n':
        if (sscanf (optarg, "%i", &min_neig) != 1)
        {
          fprintf(stderr,"ERROR: could not parse neig from %s\n", optarg);
          usage();
          return EXIT_FAILURE;
        }
        break;

      case 't':
        if (sscanf (optarg, "%f", &thresh) != 1)
        {
          fprintf(stderr,"ERROR: could not parse thresh from %s\n", optarg);
          usage();
          return EXIT_FAILURE;
        }
        break;

        case 'D':
          if (sscanf (optarg, "%s", &dbsvddb.eigs_zapped_path) != 1)
          {
            fprintf(stderr,"ERROR: could not parse in base_dir from %s\n", optarg);
            return EXIT_FAILURE;
          }
          break;

      case 'm':
        write_monitoring = 0;
        break;

      case 'd':
        daemon = 1;
        break;

      case 'v':
        verbose++;
        break;

      case 'V':
        verbose++;
        verbose++;
        break;
      
      default:
        usage();
        return 0;
    }
  }

  if (argc - optind != 2)
  {
    fprintf(stderr,"ERROR: expected 2 command line arguments\n");
    usage();
    return EXIT_FAILURE;
  }

  dbsvddb.verbose = verbose;

  if (dbsvddb.verbose)
    fprintf(stderr,"parsing input key: %s\n",argv[optind]);
  if (sscanf (argv[optind], "%x", &in_key) != 1)
  {
    fprintf(stderr, "ERROR: could not parse in key from %s\n", argv[optind]);
    return EXIT_FAILURE;
  }

  if (dbsvddb.verbose)
    fprintf(stderr,"parsing output key: %s\n",argv[optind+1]);
  if (sscanf (argv[optind+1], "%x", &dbsvddb.output.key) != 1)
  {
    fprintf(stderr,"ERROR: could not parse out key from %s\n", argv[optind+1]);
    return EXIT_FAILURE;
  }



  if (dbsvddb.nprocs < 1)
  {
    fprintf(stderr,"ERROR: invalid number of processors provided (%i)\n", 
        dbsvddb.nprocs);
    return EXIT_FAILURE;
  }
  if (dbsvddb.verbose)
    fprintf(stderr,"Using %u processors\n",dbsvddb.nprocs);

  if (dbsvddb.verbose)
  {
    fprintf(stderr,"Using min_neig: %i\n", min_neig);
    fprintf(stderr,"Using threshold: %f\n", thresh);
  }
  dbsvddb.min_neig = min_neig;
  dbsvddb.thresh = thresh;
  dbsvddb.write_monitoring = write_monitoring;
  fprintf(stderr,"WRITE MONITOR %i\n",dbsvddb.write_monitoring);

  if(dbsvddb.verbose)
  {
#if USE_DOUBLE
    fprintf(stderr,"Using double-precission float format for SVD computation\n");
#else
    fprintf(stderr,"Using single-precission float format for SVD computation\n");
#endif
  }

#ifdef mkl_malloc
  if(dbsvddb.verbose)
    fprintf(stderr,"Using INTEL MKL library\n");
#endif
  


  multilog_t* log = 0;
  log = multilog_open("mopsr_dbsvddb",0);
  multilog_add (log,stderr);



  in_hdu = dada_hdu_create(log);
  dada_hdu_set_key(in_hdu, in_key);

  if (dada_hdu_connect(in_hdu) < 0)
  {
    multilog (log, LOG_ERR, "main: could not connect to input data block (key=%x)\n",in_key);
    return EXIT_FAILURE;
  }

  

  // get the block size of the DADA data block
  uint64_t in_block_size = ipcbuf_get_bufsz( (ipcbuf_t *) in_hdu->data_block);

  // setup output data block
  dbsvddb.output.hdu = dada_hdu_create (log);
  dada_hdu_set_key (dbsvddb.output.hdu, dbsvddb.output.key);
  if (dada_hdu_connect (dbsvddb.output.hdu) <0)
  {
    multilog(log, LOG_ERR, "main: could not connect to output data block (key=%x)\n",dbsvddb.output.key);
    return EXIT_FAILURE;
  }
  if (dada_hdu_lock_read(in_hdu) < 0)
  {
    multilog(log, LOG_ERR, "main: could not lock read on hdu\n");
    return EXIT_FAILURE;
  }

  dbsvddb.output.curr_block = 0;
  dbsvddb.output.bytes_written = 0;
  dbsvddb.output.block_open = 0;
  dbsvddb.output.block_size = ipcbuf_get_bufsz( (ipcbuf_t *) dbsvddb.output.hdu->data_block);

  if (in_block_size != dbsvddb.output.block_size)
  {
    multilog(log, LOG_ERR, "Input datablock size != output datablock size; this mode is not supported. Terminating\n");
    return -1;
  }

  if (dbsvddb.verbose)
  {
    multilog(log, LOG_INFO, "dbsvddb.output.block_size=%"PRIu64"\n",
        dbsvddb.output.block_size);
    multilog(log, LOG_INFO, "in_block_size=%"PRIu64"\n",
        in_block_size);
  }

  client = dada_client_create();

  client->log               = log;
  client->data_block        = in_hdu->data_block;
  client->header_block      = in_hdu->header_block;
  client->open_function     = dbsvddb_open;
  client->io_function       = dbsvddb_write;
  client->io_block_function = dbsvddb_io_block;
  client->close_function    = dbsvddb_close;

  client->direction         = dada_client_reader;

  client->context           = &dbsvddb;
  client->quiet             = (verbose > 0) ? 0:1;

  while (!client->quit)
  {
    if (verbose)
      multilog(log, LOG_INFO, "main: dada_client_read()\n");

    if (dada_client_read (client) < 0)
      multilog(log, LOG_ERR, "main: error during transfer\n");

    if (verbose)
      multilog(log, LOG_INFO, "main: dada_hdu_unlock_read()\n");

    if (dada_hdu_unlock_read(in_hdu) <0)
    {
      multilog(log,LOG_ERR, "main: could not unlock read on hdu\n");
      return EXIT_FAILURE;
    }

    if (single_transfer || dbsvddb.quit)
      client->quit = 1;

    if (!client->quit)
    {
      if (dada_hdu_lock_read(in_hdu) < 0)
      {
        multilog(log, LOG_ERR, "main: could not lock read on hdu\n");
        return EXIT_FAILURE;
      }
    }
  }

  if (dada_hdu_disconnect (in_hdu) < 0)
  {
    multilog(log, LOG_ERR, "main: ERROR: dada_hdu_disconnect\n");
    return EXIT_FAILURE;
  }



  dada_hdu_destroy(in_hdu);
  dada_hdu_destroy(dbsvddb.output.hdu);
  multilog_close(log);
  dada_client_destroy(client);

  return EXIT_SUCCESS;
}


int get_data(char* fname, uint64_t nrows, 
    uint64_t ncols, uint64_t nchans, w_dtype** buffer)
{
  FILE* fp = fopen(fname,"r");
  size_t nread;
  nread = fread(buffer[0], sizeof(w_dtype), nrows*ncols, fp);
  if (nread != nrows*ncols)
  {
    fprintf(stderr,"ERROR: get_data. Read only %lu, expected %lu\n", 
        nread, nrows*ncols);
    return -1;
  }
  fclose(fp);

  unsigned i;
  for(i=1; i<nchans; i++)
  {
    memcpy(buffer[i], buffer[0], nrows*ncols*sizeof *buffer[0]);
  }

  return 0;
}

int dbsvddb_open(dada_client_t* client)
{
  mopsr_dbsvddb_t* ctx = (mopsr_dbsvddb_t *) client->context;

  multilog_t* log = client->log;

  char * out_header = 0;


  if (ctx->verbose)
    multilog(log, LOG_INFO, "dbsvddb_open()\n");

  if (dada_hdu_lock_write(ctx->output.hdu) < 0)
  {
    multilog (log, LOG_ERR, "dbsvddb_open: cannot lock write DADA HDU (key=%x)\n", ctx->output.key);
    return -1;
  }

  int64_t transfer_size = 0;
  ascii_header_get(client->header, "TRANSFER_SIZE", "%"PRIi64, &transfer_size);


  if (ascii_header_get (client->header, "NBEAM", "%u",&(ctx->nbeams)) != 1)
  {
    multilog(log, LOG_ERR, "open: header with no NBEAM\n");
    return -1;
  }

  if (ascii_header_get(client->header, "NCHAN", "%u", &(ctx->nchans)) != 1)
  {
    multilog(log, LOG_ERR, "open: header with no NCHAN\n");
    return -1;
  }

  if (ascii_header_get(client->header, "ORDER", "%s", &(ctx->order)) != 1)
  {
    multilog(log, LOG_ERR, "open: header with no ORDER\n");
    return -1;
  }

  if (ascii_header_get(client->header, "NBIT", "%u", &(ctx->nbit)) != 1)
  {
    multilog(log, LOG_ERR, "open: header with no NBIT\n");
    return -1;
  }

  if (ctx->nbit != 32)
  {
    multilog(log, LOG_ERR, "Unsupported nbit (%u)\n", ctx->nbit);
    return -1;
  }

  if ((strcmp(ctx->order, "SFT") != 0) && (strcmp(ctx->order, "STF") != 0))
  {
    multilog(log, LOG_ERR, "open: ORDER (%s) not supported\n", &ctx->order);
    return -1;
  }


  //if (ctx->write_monitoring)
  //{
  //  if (getcwd(eigs_zapped_path, sizeof(eigs_zapped_path)) == NULL)
  //  {
  //    multilog(log, LOG_WARNING, "open: couldn't get current working directory\n");
  //    ctx->write_monitoring = 0;
  //  }
  //}

  char utc_start[64];

  if (ctx->write_monitoring)
  {
    //if (ascii_header_get(client->header, "UTC_START", "%s", &(utc_start)) != 1)
    //{
    // multilog(log, LOG_WARNING, "open: header with no UTC_START\n");
    //  sprintf(eigs_zapped_path,"%s/monitoring.eig",eigs_zapped_path);
    //}
    //else
    //  sprintf(eigs_zapped_path,"%s/%s.eig",eigs_zapped_path,utc_start);

    ctx->eigs_zapped_file = fopen(ctx->eigs_zapped_path, "w");
    if (ctx->eigs_zapped_file == NULL)
    {
      multilog(log, LOG_ERR, "open: could not open %s for writting\n", ctx->eigs_zapped_path);
      ctx->write_monitoring = 0;
    }

    if (ctx->verbose)
      multilog(log, LOG_INFO, "open: opened %s for writting\n", ctx->eigs_zapped_path);

  }


  uint64_t block_size = ipcbuf_get_bufsz ( (ipcbuf_t *) client->data_block);

  // Expected number of samples per block
  ctx->nsamples = block_size / (ctx->nbeams * ctx->nchans * (ctx->nbit/8));

  // Number of eigenvalues/vectors to expect
  ctx->nW = MAX(1,MIN(ctx->nbeams,ctx->nsamples));


  if (ctx->verbose)
    multilog(log, LOG_INFO, "open: setting nchans=%u, nbeams=%u, nsamples=%u\n",
        ctx->nchans,ctx->nbeams,ctx->nsamples);

  // get the header from the input data block
  uint64_t header_size = ipcbuf_get_bufsz(client->header_block);

  // setupt header for output HDU
  if (ctx->verbose)
    multilog(log, LOG_INFO, "open: writing HDU %x\n", ctx->output.key);

  out_header = ipcbuf_get_next_write(ctx->output.hdu->header_block);

  if (!out_header)
  {
    multilog(log, LOG_ERR, "open: could not get next header block from HDU %x\n", ctx->output.key);
    return -1;
  }

  // copy header from input to output
  memcpy (out_header, client->header, header_size);

  // Add SVD parameters to header
  if (ascii_header_set(out_header, "SVD_THRESH", "%f", ctx->thresh) < 0)
  {
    multilog(log, LOG_ERR, "open: could not set 'SVD_THRESH' parameter (%f) in output header\n");
    return -1;
  }


  // mark the outgoing header as filled
  if (ipcbuf_mark_filled (ctx->output.hdu->header_block, header_size) < 0)
  {
    multilog(log, LOG_INFO, "open: could not mark filled header block (key=%x)\n", ctx->output.key);
    return -1;
  }

  client->transfer_bytes = transfer_size;

  ctx->bytes_in = 0;
  ctx->bytes_out = 0;
  client->header_transfer = 0;

  malloc_work(client);
  

  return 0;
}

int64_t dbsvddb_write(dada_client_t* client, void* data, uint64_t data_size)
{
  mopsr_dbsvddb_t* ctx = (mopsr_dbsvddb_t*) client->context;

  multilog_t* log = client->log;

  if (ctx->verbose)
    multilog(log,LOG_INFO, "write: t_write=%"PRIu64"\n", data_size);

  // write dat to all data blocks
  ipcio_write(ctx->output.hdu->data_block, data, data_size);

  ctx->bytes_in += data_size;
  ctx->bytes_out += data_size;

 if (ctx->verbose) 
   multilog(log, LOG_INFO, "write: read %"PRIu64", write %"PRIu64" bytes\n",
       data_size, data_size);

  return data_size;
}

int dbsvddb_close(dada_client_t* client, uint64_t bytes_written)
{
  mopsr_dbsvddb_t* ctx = (mopsr_dbsvddb_t*) client->context;

  multilog_t* log = client->log;

  if (ctx->verbose)
    multilog(log, LOG_INFO, "close: bytes_in=%"PRIu64", bytes_out=%"PRIu64"\n",
        ctx->bytes_in, ctx->bytes_out);

  if (ctx->write_monitoring)
  {
    if (ctx->verbose)
      multilog(log, LOG_INFO, "close: closing monitoring file\n");

    fclose(ctx->eigs_zapped_file);
  }

  // close the block if it is open
  if (ctx->output.block_open)
  {
    if (ctx->verbose)
      multilog(log, LOG_INFO, "close: ipcio_close_block_write\n");
    if (ipcio_close_block_write (ctx->output.hdu->data_block, ctx->output.bytes_written) < 0)
    {
      multilog(log, LOG_ERR, "dbsvddb_close: ipcio_close_block_write failed\n");
      return -1;
    }
    ctx->output.block_open = 0;
    ctx->output.bytes_written = 0;
  }

  // unlock write on the datablock (end the transfer)
  if (ctx->verbose)
    multilog(log, LOG_INFO, "close: dada_hdu_unlock_write\n");

  if (dada_hdu_unlock_write (ctx->output.hdu) < 0)
  {
    multilog(log, LOG_ERR, "dbsvddb_close: cannot unlock DADA HDU (key=%x)\n", ctx->output.key);
    return -1;
  }
  
  free_work(client);

  return 1;
}


int64_t dbsvddb_io_block(dada_client_t* client, void* in_data, uint64_t data_size, uint64_t block_id)
{
  mopsr_dbsvddb_t* ctx = (mopsr_dbsvddb_t*) client->context;

  multilog_t * log = client->log;

  if (ctx->verbose)
    multilog (log, LOG_INFO, "dbsvddb_io_block: data_size=%"PRIu64", block_id=%"PRIu64"\n",
        data_size, block_id);

  uint64_t nbeams = ctx->nbeams;
  uint64_t nchans = ctx->nchans;

  //uint64_t nsamples = ctx->nsamples; //XXX
  uint64_t nsamples = data_size / (nbeams * nchans * (ctx->nbit)/8);

  if (nsamples < MIN_SVD_NSAMPLES)
  {
    multilog(log, LOG_WARNING, "dbsvddb_io_block: too few samples (%"PRIu64") for SVD",
        nsamples);
    //return??XXX
  }

  // Depends on the order of data
  // Here, I decided I want my data to be FST
  // If you want to swap nrows and ncols, make sure you swap them in the 
  // malloc_work function as well !!!
  //ctx->nrows = ctx->nbeams;
  //ctx->ncols = ctx->nsamples;
  uint64_t nrows = ctx->nbeams;
  uint64_t ncols = nsamples;


  unsigned nproc, ichan, i,j;
  float alpha, beta;
  alpha = 1.0; beta = 0.;

  // unpack pointers, make my life easier
  w_dtype **U = ctx->work.U;
  w_dtype **Vt = ctx->work.Vt;
  w_dtype **W = ctx->work.W;
  w_dtype **superb = ctx->work.superb;
  w_dtype **PC = ctx->work.PC;
  w_dtype **data = ctx->work.data;
  w_dtype *outdata = ctx->work.outdata;
  lapack_int *info = ctx->work.info;

  lapack_int lddata, ldu, ldvt;
  lddata = ncols;
  ldu = MAX(1,MIN(ncols,nrows));
  ldvt = ncols;


  ///////////////////////////////////////////
  // This is the main data-processing loop //
  ///////////////////////////////////////////


  // TODO: Check if data_size = expected
  // if (data_size == expected_size)
  ctx->output.block_open = 1;

  if (strcmp(ctx->order, "STF") == 0)
  {
    if (ctx->verbose)
      multilog(log, LOG_INFO, "dbsvddb_io_block: transpose STF->FST\n");
    dbsvddb_transpose_STF_FST(in_data, data, nbeams, nsamples, nchans);
  }
  else if (strcmp(ctx->order,"SFT") == 0)
  {
    if (ctx->verbose)
      multilog(log, LOG_INFO, "dbsvddb_io_block: transpose SFT->FST\n");
    dbsvddb_transpose_SFT_FST(in_data, data, nbeams, nsamples, nchans);
  }
  else
  {
    multilog(log, LOG_ERR, "dbsvddb_io_block: can't handle the data ordering (%s)\n",ctx->order);
    return -1;
  }

  // Set number of processors to use
  omp_set_num_threads(ctx->nprocs);
 
  if (ctx->verbose)
    multilog(log, LOG_INFO, "Starting SVD; nrows: %lu, ncols: %lu\n",nrows,ncols);

  int neig;
  int ERR = 0;

  //FILE* fp = fopen("./eig.out","a");
  //clock_t t;
  
  char jobu,jobvt;
  if (nrows<ncols)
  {
    jobu = 'S';
    jobvt = 'N';
  }
  else
  {
    jobu = 'N';
    jobvt = 'S';
  }
  //t = clock();

#pragma omp parallel for private(nproc,i,j)
  for (ichan=0; ichan<nchans; ichan++)
  {
    nproc = omp_get_thread_num();
    if (ctx->verbose > 1)
      multilog(log, LOG_INFO, "Process %i working on ichan: %i\n",nproc, ichan);
    memcpy(&(outdata[ichan*nrows*ncols]),
        data[ichan],nrows*ncols* sizeof *data[ichan]);
    if (ERR)
      continue;

#ifdef USEDOUBLE
    info[nproc] = LAPACKE_dgesvd_work(LAPACK_ROW_MAJOR, jobu, jobvt,
        nrows, ncols, data[ichan], lddata,
        W[nproc], U[nproc], ldu, 
        Vt[nproc], ldvt, superb[nproc], ctx->work.lwork);
//    LAPACK_dgesvd(&jobu, &jobvt, &nrows, &ncols, data[ichan],
//        &lddata, W[nproc], U[nproc], &ldu, Vt[nproc],
//        &ldvt, superb[nproc], &(ctx->work.lwork), &info[nproc]);
#else
    info[nproc] = LAPACKE_sgesvd_work(LAPACK_ROW_MAJOR, jobu, jobvt,
        nrows, ncols, data[ichan], lddata,
        W[nproc], U[nproc], ldu, 
        Vt[nproc], ldvt, superb[nproc], ctx->work.lwork);
//    LAPACK_sgesvd(&jobu, &jobvt, &nrows, &ncols, data[ichan],
//        &lddata, W[nproc], U[nproc], &ldu, Vt[nproc],
//        &ldvt, superb[nproc], &(ctx->work.lwork), &info[nproc]);
#endif

    if (info[nproc] != 0)
    {
      multilog(log, LOG_ERR, "ERROR in SVD: %i, nchan: %i\n",info[nproc],ichan);
      ERR += 1;
      continue;
    }


    if (ctx->verbose > 1)
      multilog(log, LOG_INFO, "Process: %i, ichan: %i, performing SVD\n",
          nproc,ichan);
    // Compute Principal Components
    if (nrows<ncols)
    { 
    // PC = dot(U.T,data)
#ifdef USEDOUBLE
      cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
          nrows, ncols, MIN(nrows,ncols), alpha, U[nproc], ldu,
          &(outdata[ichan*nrows*ncols]),lddata,beta,
          PC[nproc],lddata);
#else
      cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
          nrows, ncols, MIN(nrows,ncols), alpha, U[nproc], ldu,
          &(outdata[ichan*nrows*ncols]),lddata,beta,
          PC[nproc],lddata);
#endif

    }
    else
    {
    // PC = dot(data,Vt.T) = dot(data,V)
#ifdef USEDOUBLE
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
          nrows, ncols, MIN(nrows,ncols), alpha, 
          &(outdata[ichan*nrows*ncols]), lddata,
          Vt[nproc],ldvt,beta,PC[nproc],lddata);
#else
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
          nrows, ncols, MIN(nrows,ncols), alpha, 
          &(outdata[ichan*nrows*ncols]), lddata,
          Vt[nproc],ldvt,beta,PC[nproc],lddata);
#endif
    }
    if (ctx->verbose > 1)
      multilog(log, LOG_INFO,"Process: %i, ichan: %i, SVD done\n",
          nproc,ichan);

//    for(i=0; i<ctx->nW; i++)
//      fprintf(stderr, "%f ", W[nproc+i]);
//    fprintf(stderr, "\n");
//    int tmptmp;
//    tmptmp = getchar ();
    // Find how many eigenvectors to zap
    neig = dbsvddb_get_neig(W[nproc], ctx->nW, ctx->min_neig, ctx->thresh);
    ctx->eigs_zapped[ichan] = neig;

    // Zeroing the first k eigenvectors (first k columns of U)
    // (Or first k columns of Vt.T, i.e. first k rows of V)
    for(i=0; i<MIN(nrows,ncols); i++)
    {
      for(j=0; j<neig; j++)
      {
        if (nrows<ncols)
        {
          U[nproc][i*MIN(nrows,ncols) + j] = 0.0;
        }
        else
        {
          Vt[nproc][j*MIN(nrows,ncols) + i] = 0.0;
        }
      }
    }

    // Projecting back
    if (nrows<ncols)
    {
    // outdata = dot(U,data)
#ifdef USEDOUBLE
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
          nrows, ncols, MIN(nrows,ncols), alpha, U[nproc], ldu,
          PC[nproc], lddata, beta, &outdata[ichan*nrows*ncols], lddata);
#else
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
          nrows, ncols, MIN(nrows,ncols), alpha, U[nproc], ldu,
          PC[nproc], lddata, beta, &outdata[ichan*nrows*ncols], lddata);
#endif
    }
    else
    {
    // outdata = dot(Y, Vt)
#ifdef USEDOUBLE
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
          nrows, ncols, MIN(nrows,ncols), alpha, PC[nproc], lddata,
          Vt[nproc], ldvt, beta, &outdata[ichan*nrows*ncols], lddata);
#else
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
          nrows, ncols, MIN(nrows,ncols), alpha, PC[nproc], lddata,
          Vt[nproc], ldvt, beta, &outdata[ichan*nrows*ncols], lddata);
#endif
    }
    //for(i=0; i<ctx->nW; i++)
    //  fprintf(fp,"%f ", W[nproc][i]);
    //fprintf(fp,"\n");

    //fprintf(stderr,"%i %f %f %f %f %f %f\n",
    //    nproc,W[nproc][0], data[ichan][0], U[nproc][0], Vt[nproc][0], PC[nproc][0],
    //    outdata[ichan*nrows*ncols+1]);

  }
  //fprintf(stderr,"\n");
  //t = clock() - t;
  //double time_taken = ((double) t)/CLOCKS_PER_SEC;
  //fprintf(stderr,"Time: %lf \n",time_taken);
  //fclose(fp);

  /* 
  // Validating
  for(i=0; i<ctx->nW; i++)
  {
    for(nproc=0; nproc<ctx->nprocs; nproc++)
    {
      fprintf(stderr,"%f ",W[nproc][i]);
    }
    fprintf(stderr,"\n");
  }
  fprintf(stderr,"\n");
  */ 

  // Write monitoring statistics
  if (ctx->write_monitoring)
    fwrite(ctx->eigs_zapped, sizeof *(ctx->eigs_zapped), nchans, ctx->eigs_zapped_file);

  // Write data to output buffer
  uint64_t out_block_id;
  ctx->output.curr_block = ipcio_open_block_write(ctx->output.hdu->data_block, &out_block_id);
  if (ctx->verbose)
    multilog(log, LOG_INFO, "dbsvddb_io_block: ipcio_open_block_write()\n");

  if (!ctx->output.curr_block)
  {
    multilog(log, LOG_ERR, 
        "dbsvd_io_block: ipcio_open_block_write() failed");
    return -1;
  }
  ctx->output.bytes_written = 0;
  

  if (strcmp(ctx->order, "STF") == 0)
  {
    if (ctx->verbose)
      multilog(log, LOG_INFO, 
          "dbsvddb_io_block: writing to output block FST->STF\n");
    dbsvddb_write_FST_STF(ctx->output.curr_block, outdata, nbeams, nsamples, nchans);
  }
  else if (strcmp(ctx->order,"SFT") == 0)
  {
    if (ctx->verbose)
      multilog(log, LOG_INFO, 
          "dbsvddb_io_block: writing to output block FST->SFT\n");
    dbsvddb_write_FST_SFT(ctx->output.curr_block, outdata, nbeams, nsamples, nchans);
  }

  ctx->output.bytes_written += data_size;

  if (ctx->verbose)
    multilog (log, LOG_INFO, 
        "dbsvddb_io_block: close_block_write written=%"PRIu64"\n", ctx->output.bytes_written);
  if (ipcio_close_block_write (ctx->output.hdu->data_block, ctx->output.bytes_written) < 0)
  {
    multilog (log, LOG_ERR, "dbsvddb_io_block: ipcio_close_block_write failed\n");
    return -1;
  }

  ctx->bytes_in += data_size;
  ctx->bytes_out += data_size;

  if (ctx->verbose > 1)
      multilog (log, LOG_INFO, "dbsvddb read %"PRIu64", wrote %"PRIu64" bytes\n", data_size, data_size);

  ctx->output.block_open = 0;\

  return (int64_t) data_size;
}

int free_work(dada_client_t* client)
{
  ////////////////////
  // Freeing memory //
  ////////////////////

  mopsr_dbsvddb_t* ctx = (mopsr_dbsvddb_t*) client->context;

  multilog_t * log = client->log;

  if (ctx->verbose)
    multilog (log, LOG_INFO, "free_work: freeing memory\n");

  FREE(ctx->eigs_zapped);

  unsigned ichan,iproc;
  for (ichan=0; ichan<ctx->nchans; ichan++)
    FREE(ctx->work.data[ichan]);

  FREE(ctx->work.data);
  FREE(ctx->work.outdata);

  for (iproc=0; iproc<ctx->nprocs; iproc++)
  {
    FREE(ctx->work.U[iproc]); FREE(ctx->work.Vt[iproc]); FREE(ctx->work.W[iproc]);
    FREE(ctx->work.superb[iproc]); FREE(ctx->work.PC[iproc]);
  }

  FREE(ctx->work.U);
  FREE(ctx->work.Vt);
  FREE(ctx->work.W);
  FREE(ctx->work.PC);
  FREE(ctx->work.superb);
  FREE(ctx->work.info);

  if (ctx->verbose)
    multilog (log, LOG_INFO, "free_work: memory freed\n");

  return 1;
}

int malloc_work(dada_client_t* client)
{

  mopsr_dbsvddb_t* ctx = (mopsr_dbsvddb_t*) client->context;

  multilog_t * log = client->log;
  uint64_t nrows = ctx->nbeams;
  uint64_t ncols = ctx->nsamples;
  unsigned int nchans = ctx->nchans;

  if (ctx->verbose)
    multilog (log, LOG_INFO, "malloc_work: Allocating memory\n");

  unsigned ichan,nproc;


  ctx->eigs_zapped = MALLOC(nchans*sizeof *(ctx->eigs_zapped));
  ctx->work.U = MALLOC(ctx->nprocs* sizeof *(ctx->work.U));
  ctx->work.Vt = MALLOC(ctx->nprocs* sizeof *(ctx->work.Vt));
  ctx->work.W = MALLOC(ctx->nprocs* sizeof *(ctx->work.W));
  ctx->work.superb = MALLOC(ctx->nprocs* sizeof *(ctx->work.superb));
  ctx->work.PC = MALLOC(ctx->nprocs* sizeof *(ctx->work.PC));
  ctx->work.info = MALLOC(ctx->nprocs* sizeof *(ctx->work.info));

  // Allocate memory for each process
  ctx->work.data = MALLOC(nchans*sizeof *(ctx->work.data));



  for (ichan=0; ichan<nchans; ichan++)
    ctx->work.data[ichan] = CALLOC(nrows*ncols,sizeof *(ctx->work.data[ichan]));

  // Output buffer
  ctx->work.outdata = MALLOC(nchans*nrows*ncols *sizeof *(ctx->work.outdata));

  for (nproc=0; nproc<ctx->nprocs; nproc++)
  {
    ctx->work.U[nproc] = CALLOC(nrows*MIN(ncols,nrows),sizeof *(ctx->work.U[nproc]));
    ctx->work.Vt[nproc] = CALLOC(MIN(ncols,nrows)*ncols,sizeof *(ctx->work.Vt[nproc]));
    ctx->work.W[nproc] = CALLOC(ctx->nW,sizeof *(ctx->work.W[nproc]));
    //ctx->work.superb[nproc] = CALLOC((MIN(nrows,ncols)),sizeof *(ctx->work.superb[nproc]));
    ctx->work.PC[nproc] = CALLOC(nrows*ncols, sizeof *(ctx->work.PC[nproc]));
  }

  lapack_int info=0;
  ctx->work.lwork = -1;
  lapack_int lddata, ldu, ldvt;
  lddata = ncols;
  ldu = MAX(1,MIN(ncols,nrows));
  ldvt = ncols;

#ifdef mkl_malloc
  if ((lddata % alignment) && (lddata > alignment))
    multilog(log, LOG_WARNING, "For better performance, ensure that lddata "
        "(%i) is divisible by the buffer alignment (%i)\n", lddata, alignment);
  if ((ldu % alignment) && (ldu > alignment))
    multilog(log, LOG_WARNING, "For better performance, ensure that ldu "
        "(%i) is divisible by the buffer alignment (%i)\n", ldu, alignment);
  if ((ldvt % alignment) && (ldvt > alignment))
    multilog(log, LOG_WARNING, "For better performance, ensure that ldvt "
        "(%i) is divisible by the buffer alignment (%i)\n", ldvt, alignment);
#endif


  // Query optimal working array size
#ifdef USEDOUBLE
  double work_query;
  info = LAPACKE_dgesvd_work(LAPACK_ROW_MAJOR, 'S', 'S',
      nrows, ncols, ctx->work.data[0], lddata,
      ctx->work.W[0], ctx->work.U[0], ldu,
      ctx->work.Vt[0], ldvt, &work_query, ctx->work.lwork);
#else
  float work_query;
  info = LAPACKE_sgesvd_work(LAPACK_ROW_MAJOR, 'S', 'S',
      nrows, ncols, ctx->work.data[0], lddata,
      ctx->work.W[0], ctx->work.U[0], ldu,
      ctx->work.Vt[0], ldvt, &work_query, ctx->work.lwork);
#endif

  ctx->work.lwork = (lapack_int) work_query;
  if (ctx->verbose)
    multilog(log, LOG_INFO, "malloc_work: requesting lwork=%i\n", ctx->work.lwork);

  for (nproc=0; nproc<ctx->nprocs; nproc++)
    ctx->work.superb[nproc] = CALLOC(ctx->work.lwork,
        sizeof *(ctx->work.superb[nproc]));

  if (ctx->verbose)
    multilog (log, LOG_INFO, "malloc_work: memory allocated\n");

  return 1;
}

int dbsvddb_transpose_STF_FST(void *ori_data, w_dtype** trans_data, 
    uint64_t nbeams, uint64_t nsamples, uint64_t nchans)
{
  io_dtype * in =  (io_dtype *) ori_data;

  int ibeam, isamp, ichan;
  for(ibeam=0; ibeam<nbeams; ibeam++)
  {
    for(isamp=0; isamp<nsamples; isamp++)
    {
      for(ichan=0; ichan<nchans; ichan++)
        trans_data[ichan][ibeam*nsamples + isamp] = 
          (w_dtype) in[ibeam*nsamples*nchans + isamp*nchans + ichan];
    }
  }
  return 1;
}

int dbsvddb_transpose_SFT_FST(void *ori_data, w_dtype** trans_data, 
    uint64_t nbeams, uint64_t nsamples, uint64_t nchans)
{
  io_dtype * in = (io_dtype *) ori_data;
  
  unsigned ibeam, isamp, ichan;
  for(ibeam=0; ibeam<nbeams; ibeam++)
  {
    for(ichan=0; ichan<nchans; ichan++)
    {
      for(isamp=0; isamp<nsamples; isamp++)
        trans_data[ichan][ibeam*nsamples + isamp] = 
           (w_dtype) in[ibeam*nchans*nsamples + ichan*nsamples + isamp];
    }
  }
  return 1;
}

int dbsvddb_write_FST_STF(char* outbuffer, w_dtype* svd_out,
    uint64_t nbeams, uint64_t nsamples, uint64_t nchans)
{
  io_dtype * out = (io_dtype*) outbuffer;
  unsigned ibeam, isamp, ichan;

  for(ibeam=0; ibeam<nbeams; ibeam++)
  {
    for(isamp=0; isamp<nsamples; isamp++)
    {
      for(ichan=0; ichan<nchans; ichan++)
      {
        out[ibeam*nsamples*nchans + isamp*nchans + ichan] = 
          (io_dtype) svd_out[ichan*nbeams*nsamples + ibeam*nsamples + isamp];
      }
    }
  }
  return 1;
}

int dbsvddb_write_FST_SFT(char* outbuffer, w_dtype* svd_out,
    uint64_t nbeams, uint64_t nsamples, uint64_t nchans)
{
  io_dtype * out = (io_dtype*) outbuffer;
  unsigned ibeam, isamp, ichan;

  for(ibeam=0; ibeam<nbeams; ibeam++)
  {
    for(ichan=0; ichan<nchans; ichan++)
    {
      for(isamp=0; isamp<nsamples; isamp++)
      {
        out[ibeam*nchans*nsamples + ichan*nsamples + isamp] = 
          (io_dtype) svd_out[ichan*nbeams*nsamples + ibeam*nsamples + isamp];
      }
    }
  }
  return 1;
}

/*
static inline int dbsvddb_get_neig(w_dtype* eigs, int total_eigs, int min_eigs, float thresh)
{
  w_dtype rmean = 0;
  w_dtype* eigs_r;
  eigs_r = malloc(total_eigs * sizeof * eigs_r);
  int neig = 0;
  int i;

  for(i=0; i<total_eigs; i++)
  {
    eigs_r[total_eigs - i - 1] = eigs[i];
  }


  // Compute running mean
  for (i=0; i<min_eigs; i++)
  {
    rmean += eigs_r[i];
  }
  rmean /= min_eigs;

  for(i=min_eigs; i<total_eigs; i++)
  {
    if (eigs_r[i]/rmean > thresh)
      neig += 1;
    else
    {
      // compute running mean
      rmean = (rmean*(i-1) + eigs_r[i])/i;
    }
  }

  free(eigs_r);
  return neig;
}
*/


static inline int dbsvddb_get_neig(w_dtype* eigs, int total_eigs, 
    int min_eigs, float thresh)
{
    int neigs_zap = 0;
    uint64_t i;
    for (i=0; i <= (total_eigs - 1); i++)
    {
      if (eigs[i]/eigs[i+1] > thresh)
        neigs_zap += 1;
    }
    return MIN(neigs_zap,min_eigs);
}
