package mw.cliquefind.io;

import mw.cliquefind.datatypes.Edge;

public final class EdgeInputFormat 
		extends RecordInputFormat<Edge> {
	
	EdgeInputFormat() throws Exception {
		super(Edge.class);
	}
}
