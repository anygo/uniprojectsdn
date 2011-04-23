package classifiers;

import statistics.*;
import exceptions.*;

public class NaiveBayes implements Classifier {
	MixtureDensity classes;
	
	boolean diag = false;
	
	public NaiveBayes(boolean diag) {
		this.diag = diag;
	}
	
	public void train(DataSet ds) throws TrainingException {
		// do a ML estimate for each class
		int nc = ds.getNumberOfClasses();
		classes = new MixtureDensity(ds.samples.get(0).x.length, nc, diag);
		for (int i = 0; i < nc; ++i) {
			classes.components[i] = Trainer.ml(ds.samplesByClass.get(i), diag);
			classes.components[i].id = i;
			classes.components[i].apr = 1./nc;
			classes.components[i].update();
		}
	}

	public int evaluate(Sample s) throws EvaluationException {
		// evaluate the mixture
		classes.evaluate(s.x);
		
		double sc = classes.components[0].lh;
		int id = 0;
		for (int i = 1; i < classes.nd; ++i) {
			if (sc < classes.components[i].lh) {
				id = i;
				sc = classes.components[i].lh;
			}
		}
		s.y = id;
		return id;
	}
	
	public int evaluate(Sample s, double [] scores) {
		classes.evaluate(s.x);
		
		scores[0] = classes.components[0].lh;
		int id = 0;
		double sum = scores[0];
		for (int i = 1; i < classes.nd; ++i) {
			scores[i] = classes.components[i].lh;
			if (scores[id] < scores[i]) 
				id = i;
			sum += scores[i];
		}
		
		for (int i = 0; i < classes.nd; ++i)
			scores[i] /= sum;
		
		s.y = id;
		return id;
	}
	
	public void evaluate(DataSet ds) throws EvaluationException {
		for (Sample s: ds.samples)
			evaluate(s);
	}

	public String toString() {
		String val = "Classifier/NaiveBayes (" + classes.nd + " classes)\n";
		for (Density d : classes.components)
			val += d.id + ": " + d + "\n";
		return val;
	}
}
