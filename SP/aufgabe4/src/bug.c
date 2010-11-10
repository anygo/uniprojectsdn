#include <stdio.h>
#include <string.h>
#include "halde.h"

int buggy() {
	int i,j;
	char* addr[6];

	for (j=0;j<30;j++) {

		for (i=0;i<5;i++) { 
			if ((addr[i]=malloc(16*sizeof(char)))==NULL) return -1;
			memset(addr[i],42,16);
		}

		strcpy(addr[0],"Peter J. Mustermann");
		strcpy(addr[1],"Im Waldweg 25");
		free(addr[2]);addr[2]="";
		strcpy(addr[3],"11111 Musterhoefen");
		strcpy(addr[4],"Deutschland");
		addr[5]="";

		printf("\n");
		printf("%s\n",addr[0]);free(addr[0]);
		printf("%s\n",addr[1]);free(addr[1]);
		printf("%s\n",addr[2]);
		printf("%s\n",addr[3]);free(addr[4]);
		printf("%s\n",addr[4]);free(addr[5]);

	}

	return 0;
}

int main(int argc, char* argv[]) {
	if (buggy()) fprintf(stderr,"fail simple test\n");
	return 0;
}
