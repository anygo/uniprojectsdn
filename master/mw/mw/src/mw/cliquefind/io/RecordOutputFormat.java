package mw.cliquefind.io;

import java.io.*;

import org.apache.hadoop.conf.*;
import org.apache.hadoop.fs.*;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.output.*;

public class RecordOutputFormat
		extends FileOutputFormat<NullWritable, Writable> {
	
	public class RecordWriter
			extends org.apache.hadoop.mapreduce.RecordWriter
			<NullWritable, Writable> {
		private DataOutputStream out_stream;

		RecordWriter(DataOutputStream out_stream) {
			this.out_stream = out_stream;
		}
	
		@Override
		public synchronized void close(TaskAttemptContext arg0) 
				throws IOException,	InterruptedException {
			out_stream.close();
		}
	
		@Override
		public synchronized void write(NullWritable a, Writable b)
				throws IOException {
			b.write(out_stream);
		}
	}

	@Override
	public org.apache.hadoop.mapreduce.RecordWriter<NullWritable, Writable> 
			getRecordWriter(TaskAttemptContext job) throws IOException {
		
		Configuration config = job.getConfiguration();
		Path file = getDefaultWorkFile(job, "");
		FileSystem fs = file.getFileSystem(config);
		FSDataOutputStream out_stream = fs.create(file, false);
		
		return new RecordWriter(out_stream);
	}
}