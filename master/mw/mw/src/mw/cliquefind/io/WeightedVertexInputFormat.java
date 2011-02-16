package mw.cliquefind.io;

import mw.cliquefind.datatypes.WeightedVertex;

public final class WeightedVertexInputFormat 
		extends RecordInputFormat<WeightedVertex> {

	WeightedVertexInputFormat() throws Exception {
		super(WeightedVertex.class);
	}

}
