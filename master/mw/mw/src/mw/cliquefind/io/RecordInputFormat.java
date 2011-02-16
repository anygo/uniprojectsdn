package mw.cliquefind.io;

import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;

import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;

public class RecordInputFormat<V extends Writable> 
		extends FileInputFormat<NullWritable, V> {

	protected final int record_size;
	protected Class<V> value_class;
	
	protected RecordInputFormat(Class<V> valueClass)
			throws Exception {
		value_class = valueClass;
		ByteArrayOutputStream ba = new ByteArrayOutputStream();
		DataOutputStream ds = new DataOutputStream(ba);
		
		V value_instance = valueClass.newInstance();

		value_instance.write(ds);
		ds.flush();
		record_size = ba.size();
	}

	public final class RecordReader
		extends org.apache.hadoop.mapreduce.RecordReader<NullWritable, V> {
		
		private V current_val;
		private FSDataInputStream inp_stream;
		private long total, remaining;
		
		@Override
		public void close() throws IOException {
			inp_stream.close();
		}
	
		@Override
		public NullWritable getCurrentKey() {
			return NullWritable.get();
		}
	
		@Override
		public V getCurrentValue() {
			return current_val;
		}
	
		@Override
		public float getProgress() {
			final long done = total - remaining;
			return ((float)done / (float)total); 
		}
	
		@Override
		public void initialize(InputSplit inp_split, 
				TaskAttemptContext task_ctx)
				throws IOException, InterruptedException {
			FileSplit split = (FileSplit) inp_split;
			Path file = split.getPath();
		    FileSystem fs = file.getFileSystem(task_ctx.getConfiguration());
		    
		    inp_stream = fs.open(file);
		    inp_stream.seek(split.getStart());
		    remaining = total = split.getLength();
		    
		    try {
		    	current_val = value_class.newInstance();
		    } catch(Throwable t) {
		    	throw new IOException("Failed to create instance", t);
		    }
		}
	
		@Override
		public boolean nextKeyValue() throws IOException {
			if (remaining == 0) return false;
			
			current_val.readFields(inp_stream);
			remaining -= record_size;
			return true;
		}
	}

	@Override
	public RecordReader createRecordReader(InputSplit inp_split,
			TaskAttemptContext task_ctx) throws IOException, 
			InterruptedException {
		return new RecordReader();
	}

	@Override
	protected long computeSplitSize(long blockSize, long minSize,
			long maxSize) {
		long val = Math.max(minSize, Math.min(maxSize, blockSize));
		
		// Make sure we are aligned to record size
		if (val < record_size) val = 0;
		val -= val % record_size;

		return val;
	}
}