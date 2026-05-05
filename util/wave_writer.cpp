/*******************************************************************
*
*    DESCRIPTION:
*      Wave file writer
*    AUTHOR:
*      ax Inc.
*    DATE:2024/05/01
*
*******************************************************************/

#include "wave_writer.h"
#include <algorithm>

#pragma pack(1)
struct FileHeader{
	char filetype[4];
	unsigned int filesize;
	char rifftype[4];
};

struct FormatHeader{
	unsigned int size;
	unsigned short id;
	unsigned short channel_n;
	unsigned int sampling_rate;
	unsigned int data_speed;
	unsigned short block_size;
	unsigned short bit_per_sample;
};
#pragma pack()

static const int BIT_PER_SAMPLE=16;
static const int FORMAT_ID=1;

#define TAG_FORMAT "fmt "
#define TAG_DATA "data"

static void create_file_header(FileHeader *fh){
	fh->filetype[0]='R';
	fh->filetype[1]='I';
	fh->filetype[2]='F';
	fh->filetype[3]='F';
	
	fh->rifftype[0]='W';
	fh->rifftype[1]='A';
	fh->rifftype[2]='V';
	fh->rifftype[3]='E';
}
	
static void create_format_header(FormatHeader *format,int data_n,int channel_n,int sampling_rate){
	format->size=sizeof(FormatHeader)-4;
	format->id=FORMAT_ID;
	format->channel_n=channel_n;
	format->sampling_rate=sampling_rate;
	format->data_speed=sampling_rate*(BIT_PER_SAMPLE/8)*channel_n;
	format->block_size=(BIT_PER_SAMPLE/8)*channel_n;
	format->bit_per_sample=BIT_PER_SAMPLE;
}

static void zero_padding(FILE *fp,int n){
	while(n>0){
		fputc(0,fp);
		n--;
	}
}

static void write_header(FILE *fp, int data_n,int channel_n,int sampling_rate){
	FileHeader fh;
	create_file_header(&fh);
	
	FormatHeader format;
	create_format_header(&format,data_n,channel_n,sampling_rate);
	
	fh.filesize=(sizeof(fh)-8)+(8+format.size)+(8+data_n*(BIT_PER_SAMPLE/8));
	fwrite(&fh,sizeof(struct FileHeader),1,fp);

	fprintf(fp,TAG_FORMAT);
	fwrite(&format,sizeof(struct FormatHeader),1,fp);

	int n=(format.size+4)-sizeof(FormatHeader);
	zero_padding(fp,n);

	fprintf(fp,TAG_DATA);
	int data_size=data_n*(BIT_PER_SAMPLE/8);
	fwrite(&data_size,4,1,fp);
}

void write_wave_file(const char *path, std::vector<float> data, int sampling_rate){
	if(path==NULL){
		return;
	}
	FILE *fp=fopen(path,"wb");
	if(fp==NULL){
		return;
	}
	int channel_n = 1;
	write_header(fp, data.size(), channel_n, sampling_rate);
	for (size_t i = 0; i < data.size(); i++){
		int pcm = data[i] * 32767;
		short pcm_16bit = std::max(std::min(pcm, 32767), -32768);
		fwrite(&pcm_16bit,2,1,fp);
	}
	fclose(fp);
}
