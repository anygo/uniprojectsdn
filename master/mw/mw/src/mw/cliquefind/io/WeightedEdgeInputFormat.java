package mw.cliquefind.io;

import mw.cliquefind.datatypes.WeightedEdge;

public final class WeightedEdgeInputFormat
		extends RecordInputFormat<WeightedEdge> {

	WeightedEdgeInputFormat() throws Exception
		{ super(WeightedEdge.class); }
}
