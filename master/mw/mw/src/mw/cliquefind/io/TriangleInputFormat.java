package mw.cliquefind.io;

import mw.cliquefind.datatypes.Triangle;

public final class TriangleInputFormat
		extends RecordInputFormat<Triangle> {

	TriangleInputFormat() throws Exception {
		super(Triangle.class);
	}
}
