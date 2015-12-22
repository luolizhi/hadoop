package org.lukey.hadoop.bayes.trainning;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IOUtils;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.InputFormat;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.JobContext;
import org.apache.hadoop.mapreduce.RecordReader;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.util.StringUtils;

public class MyInputFormat extends InputFormat<Text, Text> {

	@Override
	public List<InputSplit> getSplits(JobContext context) throws IOException, InterruptedException {
		Path[] paths = getInputPaths(context);	//paths保存的每个类别的路径，文件夹地址
		List<InputSplit> splits = new ArrayList<InputSplit>();
//		System.out.println("-6----paths = " + paths.length);
		
		for (Path path : paths) {	//每个文件夹里面的所有文件作为一个分片
//			System.out.println("path = " + path);
			FileSystem fileFS = path.getFileSystem(context.getConfiguration());
			Long len = (long) 0; // 每次重新计算分片的长度，用文件夹下面每个文件的长度之和作为分片的长度
			for (FileStatus f : fileFS.listStatus(path)) {
				len += f.getLen();
			}
			splits.add(new FileSplit(path, 0, len, null));	//没有考虑主机的信息
		}
//		System.out.println("-7----split size = " + splits.size());
		return splits;
	}

	@Override
	public RecordReader<Text, Text> createRecordReader(InputSplit split, TaskAttemptContext context)
			throws IOException, InterruptedException {
//		System.out.println("-8---CreatRecordReader---");
		MyInputRecordReader reader = new MyInputRecordReader();	//自己定义的recordReader
		reader.initialize(split, context);
		return reader;
	}

	public static Path[] getInputPaths(JobContext context) {
		String dirs = context.getConfiguration().get("mapred.input.dir", "");
		String[] list = StringUtils.split(dirs);
		Path[] result = new Path[list.length];
		for (int i = 0; i < list.length; i++) {
			result[i] = new Path(StringUtils.unEscapeString(list[i]));
		}
//		System.out.println("-5----sum_map = " + list.length);
		return result;
	}

	public static void addInputPath(Job job, Path path) throws IOException {
		//将所有的类别的文件夹路径添加为输入路径，参考FileInputFormat中的addInputPath
		Configuration conf = job.getConfiguration();
		path = path.getFileSystem(conf).makeQualified(path);
		String dirStr = StringUtils.escapeString(path.toString());

		String dirs = conf.get("mapred.input.dir");		
		conf.set("mapred.input.dir", dirs == null ? dirStr : dirs + "," + dirStr);
//		System.out.println("-4--addInputPath---ok---");
	}
}

class MyInputRecordReader extends RecordReader<Text, Text> {//定制的RecordReader
	private FileSplit filesplit;	//保存输入的分片，它将被转换成一条（key，value）记录
	private Configuration conf;		//配置对象
	private Text value = new Text();//value对象，内容为空
	private Text key = new Text();	//key对象，内容为空
	private boolean processed = false;//布尔变量记录记录是否被处理过

	@Override
	public void initialize(InputSplit split, TaskAttemptContext context) throws IOException, InterruptedException {
//		System.out.println("-9----initialize---");
		this.filesplit = (FileSplit) split;		//将输入分片强制转换成FIleSplit
		this.conf = context.getConfiguration();	//从context中获取配置信息
	}

	@Override
	public boolean nextKeyValue() throws IOException, InterruptedException {
//		System.out.println("-10----nextKeyValue---");
		if (!processed) {//如果记录没有处理过
			//从fileSplit对象获取split的字节数，创建byte数组contents
			byte[] contents = new byte[(int) filesplit.getLength()];
			Path dirPath = filesplit.getPath(); // 从fileSplit对象获取输入文件路径，整个文件夹为一个分片，这就是一个文件夹的路径
			key.set(dirPath.getName());// key设为类名即国家名
			FileSystem fs = dirPath.getFileSystem(conf);//获取文件系统对象
			FSDataInputStream in = null;//定义文件输入流对象
			FileStatus[] stats = fs.listStatus(dirPath); // 获取文件状态信息
			int read = 0;
			for (FileStatus stat : stats) {	//将所有文件的内容读取出来	
				int fileLength = (int) stat.getLen();
				Path filePath = stat.getPath();
				FileSystem fsFile = filePath.getFileSystem(conf);
				
				try {
					in = fsFile.open(filePath);	//打开文件，返回文件输入流对象
					IOUtils.readFully(in, contents, read, fileLength);//将文件内容读取到byte数组中，一起作为value值
					read += fileLength;
				} finally {
					IOUtils.closeStream(in);//关闭输入流
				}
			}
			value.set(contents, 0, contents.length);//当前文件夹中的所有内容作为value值
			processed = true;//将处理标志设为true，下次调用该方法会返回False
			return true;
		}
		return false;//如果记录处理过，返回false，表示split处理完毕
	}

	@Override
	public Text getCurrentKey() throws IOException, InterruptedException {
//		System.out.println("-12----getCurrentKey---");
		return key;
	}

	@Override
	public Text getCurrentValue() throws IOException, InterruptedException {
//		System.out.println("-13----getCurrentValue---");
		return value;
	}

	@Override
	public float getProgress() throws IOException, InterruptedException {
//		System.out.println("-11----getProgress---");
		return processed ? 1.0f : 0.0f;
	}

	@Override
	public void close() throws IOException {
//		System.out.println("-14----close---");
	}

}
