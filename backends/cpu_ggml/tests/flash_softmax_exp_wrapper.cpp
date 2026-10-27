#include MSTAR_SOFTMAX_HEADER
extern "C" void MSTAR_SOFTMAX_WRAPPER(float *p,float *c,float *s,float *m,bool *skip) {
#if MSTAR_SOFTMAX_EXP_VARIANT == 0
    mstar_flash_softmax64(p,c,s,m,skip);
#else
    mstar_flash_softmax_exp64(p,c,s,m,skip);
#endif
}
