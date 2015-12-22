package org.lukey.hadoop.bayes.trainning;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IOUtils;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.JobContext;
import org.apache.hadoop.mapreduce.RecordReader;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;

public class WholeFileInputFormat extends FileInputFormat<LongWritable, Text>{

	@Override
	public RecordReader<LongWritable, Text> createRecordReader(InputSplit split, TaskAttemptContext context)
			throws IOException, InterruptedException {
		WholeFileRecordReader reader = new WholeFileRecordReader();
		reader.initialize(split, context);
		return reader;
	}

	@Override
	protected boolean isSplitable(JobContext context, Path filename) {
		
		return false;
	}

}

class WholeFileRecordReader extends RecordReader<LongWritable, Text>{
	
	private FileSplit fileSplit;	//保存输入的分片，他将被转换成一条<key, value>记录
	private Configuration conf;		//配置对象
	private Text value = new Text();//
	private LongWritable key = new LongWritable();	//key对象，为空
	private boolean processed = false;	//布尔变量记录记录是否被处理过
	
	
	

	@Override
	public void initialize(InputSplit split, TaskAttemptContext context) throws IOException, InterruptedException {
		this.fileSplit = (FileSplit)split;		//将输入分片强制转换成fileSplit
		this.conf = context.getConfiguration();
		
	}

	@Override
	public boolean nextKeyValue() throws IOException, InterruptedException {
		if(!processed){
			byte[] contents = new byte[(int)fileSplit.getLength()];
			Path file = fileSplit.getPath();
			FileSystem fs = file.getFileSystem(conf);
			FSDataInputStream in = null;
			try{
				in = fs.open(file);
				IOUtils.readFully(in, contents, 0, contents.length);
				value.set(contents, 0, contents.length);	
			}finally{
				IOUtils.closeStream(in);
			}
			processed = true;
			return true;
		}
		return false;
	}

	@Override
	public LongWritable getCurrentKey() throws IOException, InterruptedException {
		
		return key;
	}

	@Override
	public Text getCurrentValue() throws IOException, InterruptedException {
		// TODO Auto-generated method stub
		return value;
	}

	@Override
	public float getProgress() throws IOException, InterruptedException {
		
		return processed ? 1.0f : 0.0f;
	}

	@Override
	public void close() throws IOException {
		// TODO Auto-generated method stub
		
	}
	
	
	
	
}

