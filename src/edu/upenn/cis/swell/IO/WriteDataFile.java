package edu.upenn.cis.swell.IO;

/**
 * ver: 1.0
 * @author paramveer dhillon.
 *
 * last modified: 09/04/13
 * please send bug reports and suggestions to: dhillon@cis.upenn.edu
 */


import java.util.ArrayList;
import java.util.HashMap;

import edu.upenn.cis.swell.SpectralRepresentations.CCARepresentation;
import edu.upenn.cis.swell.SpectralRepresentations.ContextPCANGramsRepresentation;

public class WriteDataFile {
	
	
	Options _opt;
	
	ArrayList<ArrayList<Integer>> _allDocs;
	
	HashMap<Double, Integer> _allDocsH;
	
	Object[] _allDocsO;
	
	public WriteDataFile(Options opt,ArrayList<ArrayList<Integer>> all_Docs){
		_opt=opt;
		_allDocs=all_Docs;
	}
	

	public WriteDataFile(Options opt, HashMap<Double, Integer> all_Docs) {
		_opt=opt;
		_allDocsH=all_Docs;
	}
	
	public WriteDataFile(Options opt, Object[] all_Docs) {
		_opt=opt;
		_allDocsO=all_Docs;
	}

	public WriteDataFile(Options opt, CCARepresentation ccaRep) {
		_opt=opt;
		_allDocs=ccaRep.getAllDocs();
	}


	public WriteDataFile(Options opt) {
		_opt=opt;
	}
		
	
}
