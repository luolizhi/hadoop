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

import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.JobContext;
import org.apache.hadoop.mapreduce.RecordReader;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.mapreduce.lib.input.CombineFileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.util.StringUtils;

public class PredictInputFormat extends CombineFileInputFormat<Text, Text> {
	@Override
	public List<InputSplit> getSplits(JobContext context) throws IOException {
		Path[] paths = getInputPaths(context); // paths保存的每个类别的路径，文件夹地址
		List<InputSplit> splits = new ArrayList<InputSplit>();
		for (Path path : paths) { // 每个文件夹里面的所有文件作为一个分片
			FileSystem fileFS = path.getFileSystem(context.getConfiguration());
			Long len = (long) 0; 
			for (FileStatus f : fileFS.listStatus(path)) {
				len += f.getLen();
			}
			splits.add(new FileSplit(path, 0, len, null)); // 没有考虑主机的信息
		}//CombineFileSplit(paths,0,len,null)
		return splits;
	}

	@Override
	public RecordReader<Text, Text> createRecordReader(InputSplit split, TaskAttemptContext context)
			throws IOException {
		PredictInputRecordReader reader = new PredictInputRecordReader(); // 自己定义的recordReader
		try {
			reader.initialize(split, context);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
		return reader;
	}

	public static Path[] getInputPaths(JobContext context) {
		String dirs = context.getConfiguration().get("mapred.input.dir", "");
		String[] list = StringUtils.split(dirs);
		Path[] result = new Path[list.length];
		for (int i = 0; i < list.length; i++) {
			result[i] = new Path(StringUtils.unEscapeString(list[i]));
		}
		return result;
	}

	public static void addInputPath(Job job, Path path) throws IOException {
		// 将所有的类别的文件夹路径添加为输入路径，参考FileInputFormat中的addInputPath
		Configuration conf = job.getConfiguration();
		path = path.getFileSystem(conf).makeQualified(path);
		String dirStr = StringUtils.escapeString(path.toString());
		String dirs = conf.get("mapred.input.dir");
		conf.set("mapred.input.dir", dirs == null ? dirStr : dirs + "," + dirStr);
	}
}

class PredictInputRecordReader extends RecordReader<Text, Text> {// 定制的RecordReader
	private FileSplit filesplit; // 保存输入的分片，它将被转换成一条（key，value）记录
	private Configuration conf; // 配置对象
	private Text value = new Text();// value对象，内容为空
	private Text key = new Text(); // key对象，内容为空

	private int index;
	private int length;

	@Override
	public void initialize(InputSplit split, TaskAttemptContext context) throws IOException, InterruptedException {
		this.filesplit = (FileSplit) split; // 将输入分片强制转换成FIleSplit
		this.conf = context.getConfiguration(); // 从context中获取配置信息
		Path dirPath = filesplit.getPath();
		length = dirPath.getFileSystem(conf).listStatus(dirPath).length;
		index = 0;
	}

	@Override
	public boolean nextKeyValue() throws IOException, InterruptedException {
		if (length == 0 || index == length) {
			return false;
		}
		Path dirPath = filesplit.getPath(); // 从fileSplit对象获取输入文件路径，整个文件夹为一个分片，这就是一个文件夹的路径
		key.set(dirPath.getName());// key设为类名即国家名
		FileSystem fs = dirPath.getFileSystem(conf);// 获取文件系统对象
		FSDataInputStream in = null;// 定义文件输入流对象
		FileStatus[] stats = fs.listStatus(dirPath); // 获取文件状态信息
		Path file = stats[index].getPath();
		key.set(file.getParent().getName() + "\t" + file.getName());// 类别+文件名
//		System.out.println("----key="+key);
		int fileLength = (int) stats[index].getLen();
		FileSystem fsFile = file.getFileSystem(conf);
		byte[] contents = new byte[fileLength];// 从fileSplit对象获取split的字节数，创建byte数组contents
		try {
			in = fsFile.open(file); // 打开文件，返回文件输入流对象
			IOUtils.readFully(in, contents, 0, fileLength);// 将文件内容读取到byte数组中，一起作为value值
		} finally {
			IOUtils.closeStream(in);// 关闭输入流
		}
		value.set(contents, 0, contents.length);// 当前文件夹中的所有内容作为value值
		index++;//没有自增运算会导致死循环
		return true;

	}

	@Override
	public Text getCurrentKey() throws IOException, InterruptedException {
		return key;
	}

	@Override
	public Text getCurrentValue() throws IOException, InterruptedException {
		return value;
	}

	@Override
	public float getProgress() throws IOException, InterruptedException {
		if (length == 0){
			return 0.0f;
		}else
			return (float) Math.min((float)index/length, 1.0);
	}

	@Override
	public void close() throws IOException {
	}

}
