package edu.upenn.cis.swell.IO;

/**
 * ver: 1.0
 * @author paramveer dhillon.
 *
 * last modified: 09/04/13
 * please send bug reports and suggestions to: dhillon@cis.upenn.edu
 */


import java.io.IOException;

import Jama.Matrix;

public interface EmbeddingWriter {
	
	public void writeEigenDict()  throws IOException;
	
	public void writeContextObliviousEmbed(Matrix contextObliviousEmbed)  throws IOException;

}
