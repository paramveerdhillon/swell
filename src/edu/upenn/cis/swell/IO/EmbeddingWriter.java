package edu.upenn.cis.swell.IO;

import java.io.IOException;

import Jama.Matrix;

public interface EmbeddingWriter {
	
	public void writeEigenDict()  throws IOException;
	
	public void writeContextObliviousEmbed(Matrix contextObliviousEmbed)  throws IOException;

}
