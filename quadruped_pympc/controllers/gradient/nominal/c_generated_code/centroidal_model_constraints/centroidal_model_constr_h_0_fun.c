/* This file was automatically generated by CasADi 3.6.5.
 *  It consists of: 
 *   1) content generated by CasADi runtime: not copyrighted
 *   2) template code copied from CasADi source: permissively licensed (MIT-0)
 *   3) user code: owned by the user
 *
 */
#ifdef __cplusplus
extern "C" {
#endif

/* How to prefix internal symbols */
#ifdef CASADI_CODEGEN_PREFIX
  #define CASADI_NAMESPACE_CONCAT(NS, ID) _CASADI_NAMESPACE_CONCAT(NS, ID)
  #define _CASADI_NAMESPACE_CONCAT(NS, ID) NS ## ID
  #define CASADI_PREFIX(ID) CASADI_NAMESPACE_CONCAT(CODEGEN_PREFIX, ID)
#else
  #define CASADI_PREFIX(ID) centroidal_model_constr_h_0_fun_ ## ID
#endif

#include <math.h>

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int int
#endif

/* Add prefix to internal symbols */
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)
#define casadi_s2 CASADI_PREFIX(s2)
#define casadi_s3 CASADI_PREFIX(s3)
#define casadi_s4 CASADI_PREFIX(s4)

/* Symbol visibility in DLLs */
#ifndef CASADI_SYMBOL_EXPORT
  #if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
    #if defined(STATIC_LINKED)
      #define CASADI_SYMBOL_EXPORT
    #else
      #define CASADI_SYMBOL_EXPORT __declspec(dllexport)
    #endif
  #elif defined(__GNUC__) && defined(GCC_HASCLASSVISIBILITY)
    #define CASADI_SYMBOL_EXPORT __attribute__ ((visibility ("default")))
  #else
    #define CASADI_SYMBOL_EXPORT
  #endif
#endif

static const casadi_int casadi_s0[34] = {30, 1, 0, 30, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29};
static const casadi_int casadi_s1[28] = {24, 1, 0, 24, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};
static const casadi_int casadi_s2[3] = {0, 0, 0};
static const casadi_int casadi_s3[33] = {29, 1, 0, 29, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28};
static const casadi_int casadi_s4[24] = {20, 1, 0, 20, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19};

/* centroidal_model_constr_h_0_fun:(i0[30],i1[24],i2[],i3[29])->(o0[20]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a2, a3, a4, a5;
  a0=arg[1]? arg[1][12] : 0;
  a1=arg[3]? arg[3][4] : 0;
  a2=arg[1]? arg[1][14] : 0;
  a3=(a1*a2);
  a4=(a0-a3);
  if (res[0]!=0) res[0][0]=a4;
  a4=arg[1]? arg[1][13] : 0;
  a5=(a4-a3);
  if (res[0]!=0) res[0][1]=a5;
  a4=(a4+a3);
  if (res[0]!=0) res[0][2]=a4;
  a0=(a0+a3);
  if (res[0]!=0) res[0][3]=a0;
  if (res[0]!=0) res[0][4]=a2;
  a2=arg[1]? arg[1][15] : 0;
  a0=arg[1]? arg[1][17] : 0;
  a3=(a1*a0);
  a4=(a2-a3);
  if (res[0]!=0) res[0][5]=a4;
  a4=arg[1]? arg[1][16] : 0;
  a5=(a4-a3);
  if (res[0]!=0) res[0][6]=a5;
  a4=(a4+a3);
  if (res[0]!=0) res[0][7]=a4;
  a2=(a2+a3);
  if (res[0]!=0) res[0][8]=a2;
  if (res[0]!=0) res[0][9]=a0;
  a0=arg[1]? arg[1][18] : 0;
  a2=arg[1]? arg[1][20] : 0;
  a3=(a1*a2);
  a4=(a0-a3);
  if (res[0]!=0) res[0][10]=a4;
  a4=arg[1]? arg[1][19] : 0;
  a5=(a4-a3);
  if (res[0]!=0) res[0][11]=a5;
  a4=(a4+a3);
  if (res[0]!=0) res[0][12]=a4;
  a0=(a0+a3);
  if (res[0]!=0) res[0][13]=a0;
  if (res[0]!=0) res[0][14]=a2;
  a2=arg[1]? arg[1][21] : 0;
  a0=arg[1]? arg[1][23] : 0;
  a1=(a1*a0);
  a3=(a2-a1);
  if (res[0]!=0) res[0][15]=a3;
  a3=arg[1]? arg[1][22] : 0;
  a4=(a3-a1);
  if (res[0]!=0) res[0][16]=a4;
  a3=(a3+a1);
  if (res[0]!=0) res[0][17]=a3;
  a2=(a2+a1);
  if (res[0]!=0) res[0][18]=a2;
  if (res[0]!=0) res[0][19]=a0;
  return 0;
}

CASADI_SYMBOL_EXPORT int centroidal_model_constr_h_0_fun(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int centroidal_model_constr_h_0_fun_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int centroidal_model_constr_h_0_fun_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void centroidal_model_constr_h_0_fun_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int centroidal_model_constr_h_0_fun_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void centroidal_model_constr_h_0_fun_release(int mem) {
}

CASADI_SYMBOL_EXPORT void centroidal_model_constr_h_0_fun_incref(void) {
}

CASADI_SYMBOL_EXPORT void centroidal_model_constr_h_0_fun_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int centroidal_model_constr_h_0_fun_n_in(void) { return 4;}

CASADI_SYMBOL_EXPORT casadi_int centroidal_model_constr_h_0_fun_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_real centroidal_model_constr_h_0_fun_default_in(casadi_int i) {
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* centroidal_model_constr_h_0_fun_name_in(casadi_int i) {
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* centroidal_model_constr_h_0_fun_name_out(casadi_int i) {
  switch (i) {
    case 0: return "o0";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* centroidal_model_constr_h_0_fun_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    case 2: return casadi_s2;
    case 3: return casadi_s3;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* centroidal_model_constr_h_0_fun_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s4;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int centroidal_model_constr_h_0_fun_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 4;
  if (sz_res) *sz_res = 1;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}

CASADI_SYMBOL_EXPORT int centroidal_model_constr_h_0_fun_work_bytes(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 4*sizeof(const casadi_real*);
  if (sz_res) *sz_res = 1*sizeof(casadi_real*);
  if (sz_iw) *sz_iw = 0*sizeof(casadi_int);
  if (sz_w) *sz_w = 0*sizeof(casadi_real);
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
