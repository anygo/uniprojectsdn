package mw.cliquefind;

import java.io.IOException;

import mw.cliquefind.S1CalculateVertexDegreesJob.FriendCount;
import mw.cliquefind.S4ListTriangleCandidatesJob.TriangleCountS4;
import mw.cliquefind.S5FindTrianglesJob.TriangleCountS5;
import mw.cliquefind.S6FindTrussJob.MyGroup;
import mw.cliquefind.io.EdgeInputFormat;
import mw.cliquefind.io.MultipleInputs;
import mw.cliquefind.io.TriangleInputFormat;
import mw.cliquefind.io.WeightedVertexInputFormat;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class FindTrussIteration {
	
	public  void run(Configuration conf) throws Exception {
		
		int i = 0;
		while (true) {
		
			String graph;
			if (i++ == 0)
				graph = conf.get("input");
			else
				graph = conf.get("output");
			
			// Job 1
			Job job = new S1CalculateVertexDegreesJob(conf, "calculateVertexdDegrees");
			
			FileInputFormat.setInputPaths(job, graph);
			FileOutputFormat.setOutputPath(job, new Path("/tmp/ausgabe_S1"));
			
			job.submit();
			job.waitForCompletion(conf.getBoolean("verbose", false));
			long numFS = job.getCounters().findCounter(FriendCount.COUNT_FRIENDSHIPS).getValue(); 
			long numNodes = job.getCounters().findCounter(FriendCount.COUNT_NODES).getValue(); 
			double avgNumFriends = (double)numFS / (double)numNodes;
			System.out.println("Job 1 abgeschlossen! + avgNumFriends: " + avgNumFriends + " " + numFS + " " + numNodes);
			
			
			// Job 2
			job = new S2JoinVertexDegreesJob(conf, "joinVertex");
			
			MultipleInputs.addInputPath(job, new Path(graph), EdgeInputFormat.class, S2JoinVertexDegreesJob.MapEdge.class);
			MultipleInputs.addInputPath(job, new Path("/tmp/ausgabe_S1"), WeightedVertexInputFormat.class, S2JoinVertexDegreesJob.MapWeightedVertex.class);
	
			FileOutputFormat.setOutputPath(job, new Path("/tmp/ausgabe_S2"));
			
			job.submit();
			job.waitForCompletion(false);
			
			System.out.println("Job 2 abgeschlossen! ");
			delete(conf, "/tmp/ausgabe_S1");
			
			
			// Job 3
			job = new S3GraphDirectionJob(conf, "graphdDirection");
			
			FileInputFormat.setInputPaths(job, "/tmp/ausgabe_S2");
			FileOutputFormat.setOutputPath(job, new Path("/tmp/ausgabe_S3"));
			
			job.submit();
			job.waitForCompletion(false);
			
			System.out.println("Job 3 abgeschlossen!");
			delete(conf, "/tmp/ausgabe_S2");
			
			
			// Job 4
			job = new S4ListTriangleCandidatesJob(conf, "listTriangleCandidates");
			
			FileInputFormat.setInputPaths(job, "/tmp/ausgabe_S3");
			FileOutputFormat.setOutputPath(job, new Path("/tmp/ausgabe_S4"));
			
			job.submit();
			job.waitForCompletion(false);
			long potentialTriangles = job.getCounters().findCounter(TriangleCountS4.COUNT_TRIANGLES).getValue();
			
			System.out.println("Job 4 abgeschlossen! " + job.getCounters().findCounter(TriangleCountS4.COUNT_MAPCALLS).getValue());
			delete(conf, "/tmp/ausgabe_S3");
			
			
			// Job 5
			job = new S5FindTrianglesJob(conf, "findTriangles");
			
			MultipleInputs.addInputPath(job, new Path(graph), EdgeInputFormat.class, S5FindTrianglesJob.MapEdge.class);
			MultipleInputs.addInputPath(job, new Path("/tmp/ausgabe_S4"), TriangleInputFormat.class, S5FindTrianglesJob.MapTriangle.class);
	
			FileOutputFormat.setOutputPath(job, new Path("/tmp/ausgabe_S5"));
			
			job.submit();
			job.waitForCompletion(false);
			long passedTriangles = job.getCounters().findCounter(TriangleCountS5.COUNT_TRIANGLES).getValue();
			
			System.out.println("Job 5 abgeschlossen! " + passedTriangles + " of " + potentialTriangles + " Triangles remainin");
			delete(conf, "/tmp/ausgabe_S4");
			
			
			// Job6
			job = new S6FindTrussJob(conf, "findTruss");
			
			if (i != 0) 
				delete(conf, conf.get("output"));
			
			
			FileInputFormat.setInputPaths(job, "/tmp/ausgabe_S5");
			FileOutputFormat.setOutputPath(job, new Path(conf.get("output")));
			
			job.submit();
			job.waitForCompletion(false);
			
			System.out.println("Job 6 abgeschlossen! " + job.getCounters().findCounter(MyGroup.WEGGEFALLEN_COUNT).getValue() + "Kanten weggefallen");
			delete(conf, "/tmp/ausgabe_S5");
			
			if (job.getCounters().findCounter(MyGroup.WEGGEFALLEN_COUNT).getValue() == 0) {
				break;
			}
		}
	}
	
	
	// delete generated files
	public void delete(Configuration conf, String name) {
		Path p = new Path(name);
		try {
			FileSystem fs = p.getFileSystem(conf);
			fs.delete(p, true);
		} catch (IOException e) {
			System.err.println("Delete failed for " + name);
		}
	}
	
}
