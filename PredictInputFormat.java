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
		Path[] paths = getInputPaths(context); // paths�����ÿ������·�����ļ��е�ַ
		List<InputSplit> splits = new ArrayList<InputSplit>();
		for (Path path : paths) { // ÿ���ļ�������������ļ���Ϊһ����Ƭ
			FileSystem fileFS = path.getFileSystem(context.getConfiguration());
			Long len = (long) 0; 
			for (FileStatus f : fileFS.listStatus(path)) {
				len += f.getLen();
			}
			splits.add(new FileSplit(path, 0, len, null)); // û�п�����������Ϣ
		}//CombineFileSplit(paths,0,len,null)
		return splits;
	}

	@Override
	public RecordReader<Text, Text> createRecordReader(InputSplit split, TaskAttemptContext context)
			throws IOException {
		PredictInputRecordReader reader = new PredictInputRecordReader(); // �Լ������recordReader
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
		// �����е������ļ���·�����Ϊ����·�����ο�FileInputFormat�е�addInputPath
		Configuration conf = job.getConfiguration();
		path = path.getFileSystem(conf).makeQualified(path);
		String dirStr = StringUtils.escapeString(path.toString());
		String dirs = conf.get("mapred.input.dir");
		conf.set("mapred.input.dir", dirs == null ? dirStr : dirs + "," + dirStr);
	}
}

class PredictInputRecordReader extends RecordReader<Text, Text> {// ���Ƶ�RecordReader
	private FileSplit filesplit; // ��������ķ�Ƭ��������ת����һ����key��value����¼
	private Configuration conf; // ���ö���
	private Text value = new Text();// value��������Ϊ��
	private Text key = new Text(); // key��������Ϊ��

	private int index;
	private int length;

	@Override
	public void initialize(InputSplit split, TaskAttemptContext context) throws IOException, InterruptedException {
		this.filesplit = (FileSplit) split; // �������Ƭǿ��ת����FIleSplit
		this.conf = context.getConfiguration(); // ��context�л�ȡ������Ϣ
		Path dirPath = filesplit.getPath();
		length = dirPath.getFileSystem(conf).listStatus(dirPath).length;
		index = 0;
	}

	@Override
	public boolean nextKeyValue() throws IOException, InterruptedException {
		if (length == 0 || index == length) {
			return false;
		}
		Path dirPath = filesplit.getPath(); // ��fileSplit�����ȡ�����ļ�·���������ļ���Ϊһ����Ƭ�������һ���ļ��е�·��
		key.set(dirPath.getName());// key��Ϊ������������
		FileSystem fs = dirPath.getFileSystem(conf);// ��ȡ�ļ�ϵͳ����
		FSDataInputStream in = null;// �����ļ�����������
		FileStatus[] stats = fs.listStatus(dirPath); // ��ȡ�ļ�״̬��Ϣ
		Path file = stats[index].getPath();
		key.set(file.getParent().getName() + "\t" + file.getName());// ���+�ļ���
//		System.out.println("----key="+key);
		int fileLength = (int) stats[index].getLen();
		FileSystem fsFile = file.getFileSystem(conf);
		byte[] contents = new byte[fileLength];// ��fileSplit�����ȡsplit���ֽ���������byte����contents
		try {
			in = fsFile.open(file); // ���ļ��������ļ�����������
			IOUtils.readFully(in, contents, 0, fileLength);// ���ļ����ݶ�ȡ��byte�����У�һ����Ϊvalueֵ
		} finally {
			IOUtils.closeStream(in);// �ر�������
		}
		value.set(contents, 0, contents.length);// ��ǰ�ļ����е�����������Ϊvalueֵ
		index++;//û����������ᵼ����ѭ��
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
