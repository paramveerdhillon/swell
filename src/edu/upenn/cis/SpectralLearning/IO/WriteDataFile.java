package edu.upenn.cis.SpectralLearning.IO;

import java.util.ArrayList;
import java.util.HashMap;

import edu.upenn.cis.SpectralLearning.Data.Corpus;
import edu.upenn.cis.SpectralLearning.SpectralRepresentations.CCARepresentation;
import edu.upenn.cis.SpectralLearning.SpectralRepresentations.ContextPCANGramsRepresentation;

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
