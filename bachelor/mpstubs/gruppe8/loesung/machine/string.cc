
extern "C" {

    void* memcpy(void *dest, const void *src, unsigned long n) {
	for(char* d = (char *) dest, * s = (char*) src; d < ((char*) dest) + n; ++d, ++s) {
	    *d = *s; 
	} 	
	return dest;
    }


}

