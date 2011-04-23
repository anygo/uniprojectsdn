package classifiers;

import statistics.*;
import exceptions.*;

public class MixtureClassifier implements Classifier {
		
	MixtureDensity [] classes;
	
	public void train(DataSet ds) 
		throws TrainingException {
		
		train(ds, 2, 5);
	}
	
	public void train(DataSet ds, int nd, int iters) 
		throws TrainingException {
		
		// do a EM estimate for each class
		int nc = ds.getNumberOfClasses();
		classes = new MixtureDensity [nc];
		for (int i = 0; i < nc; ++i) {
			MixtureDensity initial = Initialization.kMeansClustering(ds.samplesByClass.get(i), nd, true);
			classes[i] = Trainer.em(initial, ds.samplesByClass.get(i), iters);
			classes[i].id = i;
		}
	}

	public int evaluate(Sample s) 
		throws EvaluationException {
		
		classes[0].llh = 0.;
		classes[0].evaluate(s.x);
		
		double sc = classes[0].llh;
		int id = classes[0].id;
		
		for (int i = 1; i < classes.length; ++i) {
			classes[i].llh = 0;
			classes[i].evaluate(s.x);
			
			if (sc < classes[i].llh) {
				id = i;
				sc = classes[i].llh;
			}
		}
		
		s.y = id;
		return id;
	}
	
	public int evaluate(Sample s, double [] scores) {
		classes[0].llh = 0;
		classes[0].evaluate(s.x);
		
		scores[0] = classes[0].llh;
		int id = classes[0].id;
		double sum = scores[0];
		
		for (int i = 1; i < classes.length; ++i) {
			classes[i].llh = 0.;
			classes[i].evaluate(s.x);
			scores[i] = classes[i].llh;
			sum += scores[i];
			
			if (scores[id] < scores[i])
				id = i;
				
		}
		
		for (int i = 0; i < scores.length; ++i)
			scores[i] /= sum;
		
		s.y = id;
		return id;
	}
	
	public void evaluate(DataSet ds) 
		throws EvaluationException {
		
		for (Sample s: ds.samples)
			evaluate(s);
	}

	public String toString() {
		String val = "Classifier/MixtureClassifier (" + classes.length + " classes)\n";
		for (MixtureDensity d : classes)
			val += d.id + ": " + d;
		return val;
	}
}
