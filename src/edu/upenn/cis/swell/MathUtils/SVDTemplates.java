package edu.upenn.cis.swell.MathUtils;

/**
 * ver: 1.0
 * @author paramveer dhillon.
 *
 * last modified: 09/04/13
 * please send bug reports and suggestions to: dhillon@cis.upenn.edu
 */


import java.io.Serializable;
import Jama.Matrix;
import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.QR;
import cern.colt.matrix.tdouble.algo.DenseDoubleAlgebra;
import cern.colt.matrix.tdouble.algo.decomposition.DenseDoubleSingularValueDecomposition;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix2D;
import cern.colt.matrix.tdouble.impl.SparseDoubleMatrix2D;
import edu.upenn.cis.swell.IO.Options;

public class SVDTemplates implements Serializable {

	Options _opt;
	int _dimension2=0;
	private DenseDoubleMatrix2D dictMatrixCOLT=null;
	long startTime=0,endTime=0;
	static final long serialVersionUID = 42L;
	double[] sVals;
	
	public SVDTemplates(Options opt, int numDocs){
		this(opt);
		_dimension2=numDocs;
		
	}
	
	public SVDTemplates(Options opt){
		_opt=opt;
		
	}
	
	public SparseDoubleMatrix2D computeSparseInverse(SparseDoubleMatrix2D X){
		
		//Computes inverse of diagonally dominant sparse matrices by power series expansion as
		// (A+ eB)^-1 = A^-1 - e. A^-1.B.A^-1 + e^2.A^-1.B.A^-1.B.A^-1...Below we only use the first two terms.
		//In our case A is a diagonal matrix so its inverse is easy and B is the remainder. 
		
		
		
		//SparseDoubleMatrix2D tempMat=new SparseDoubleMatrix2D(X.rows(),X.columns(),0,0.7,0.75);
		//SparseDoubleMatrix2D auxMat=new SparseDoubleMatrix2D(X.rows(),X.columns(),0,0.7,0.75);
		SparseDoubleMatrix2D diagInvEntries=new SparseDoubleMatrix2D(X.rows(),X.columns(),0,0.7,0.75);
		//SparseDoubleMatrix2D OffdiagEntries=new SparseDoubleMatrix2D(X.rows(),X.columns(),0,0.7,0.75);
		
		
		System.out.println("++Beginning Sparse Inverse++");
		
		
		for(int i=0; i<X.rows();i++){ 
			
			//if(X.get(i, i)!=0){
				diagInvEntries.set(i, i, 1/X.get(i, i));
			//}else{
				//diagInvEntries.set(i, i, 1);
			//}
		}
		System.out.println("++Finished Sparse Inverse++");
		
		/*
		if(!_opt.diagOnlyInverse){
		
			for(int i=0; i<X.rows();i++){ 
				for(int j=0; j<X.columns();j++){ 
					if(i!=j){
						OffdiagEntries.set(i, j, X.get(i, j));
							}
												}
								}
			
			diagInvEntries.zMult(OffdiagEntries, tempMat);
			tempMat.zMult(diagInvEntries, auxMat);
	
			auxMat.assign(diagInvEntries, DoublePlusMultFirst.minusMult(1));
			return auxMat;
		}
		else{
		*/
		
			return diagInvEntries;
		//}
		
	}
	
	public Matrix computeSVD_Tropp(SparseDoubleMatrix2D X, DenseDoubleMatrix2D omega)
	{
		dictMatrixCOLT=new DenseDoubleMatrix2D(X.rows(),_opt.hiddenStateSize);
		DenseDoubleMatrix2D Xomega=new DenseDoubleMatrix2D(X.rows(),_opt.hiddenStateSize+20);//Oversample the required rank.
		DenseDoubleMatrix2D UhatTemp=new DenseDoubleMatrix2D(_opt.hiddenStateSize+20,_opt.hiddenStateSize+20);
		DenseDoubleMatrix2D UhatTemp1=new DenseDoubleMatrix2D(_opt.hiddenStateSize+20,_opt.hiddenStateSize+20);
		
		DenseDoubleMatrix2D Uhat=new DenseDoubleMatrix2D(_opt.hiddenStateSize+20,_opt.hiddenStateSize);
		
		SparseDoubleMatrix2D sValsOmega=new SparseDoubleMatrix2D(_opt.hiddenStateSize+20,_opt.hiddenStateSize+20);
		DenseDoubleMatrix2D b=new DenseDoubleMatrix2D(_opt.hiddenStateSize+20,_dimension2);
		DenseDoubleMatrix2D q=new DenseDoubleMatrix2D(X.rows(),_opt.hiddenStateSize+20);

		for(int powIter =0; powIter <5;powIter++){
			startTime = System.currentTimeMillis();
			X.zMult(omega, Xomega);
			endTime = System.currentTimeMillis();
		//System.out.println("===Time taken for Multiplication: "+(endTime-startTime)/1000.0/60.0+" mins===");

			startTime = System.currentTimeMillis();

			QR qr=new QR(Xomega.rows(),Xomega.columns());


			DenseMatrix XomegaMTJ=MatrixFormatConversion.createDenseMatrixMTJ(Xomega);
			DenseMatrix qMTJ=qr.factor(XomegaMTJ).getQ();
			q=MatrixFormatConversion.createDenseMatrixCOLT(qMTJ);

			DenseDoubleAlgebra dalg=new DenseDoubleAlgebra();
			DenseDoubleMatrix2D qt=(DenseDoubleMatrix2D) dalg.transpose(q);

			qt.zMult(X, b);
			omega= (DenseDoubleMatrix2D)dalg.transpose(b);
			endTime = System.currentTimeMillis();
		}
		//System.out.println("===Time taken for QR and multiply: "+(endTime-startTime)/1000.0/60.0+" mins===");

		startTime = System.currentTimeMillis();

		DenseDoubleSingularValueDecomposition svd= new DenseDoubleSingularValueDecomposition(b,true,false);
		
		
		UhatTemp1=(DenseDoubleMatrix2D) svd.getU();
		
		if(_opt.scaleBySingVals){
			sValsOmega=(SparseDoubleMatrix2D) svd.getS();
			sValsOmega.normalize();
			UhatTemp1.zMult(sValsOmega, UhatTemp);
		}
		else{
			UhatTemp=UhatTemp1;
		}
		
		
		for(int i=0; i<_opt.hiddenStateSize+20;i++){ //Take only the top k elements of the matrix after svd.
			for (int j=0;j<_opt.hiddenStateSize;j++){
				Uhat.set(i, j, UhatTemp.get(i, j));
			}
		}
		
		endTime = System.currentTimeMillis();
		System.out.println("===Time taken for SVD: "+(endTime-startTime)/1000.0/60.0+" mins===");




		startTime = System.currentTimeMillis();
		q.zMult(Uhat, dictMatrixCOLT);
		endTime = System.currentTimeMillis();
		System.out.println("===Time taken for Final Multiply: "+(endTime-startTime)/1000.0/60.0+" mins===");

		return MatrixFormatConversion.createDenseMatrixJAMA(dictMatrixCOLT);
	
}
	
	
	public Matrix computeSVD_Tropp(SparseDoubleMatrix2D X, DenseDoubleMatrix2D omega, int _dim2)
	{
		dictMatrixCOLT=new DenseDoubleMatrix2D(X.rows(),_opt.hiddenStateSize);
		DenseDoubleMatrix2D Xomega=new DenseDoubleMatrix2D(X.rows(),_opt.hiddenStateSize+20);//Oversample the required rank.
		DenseDoubleMatrix2D UhatTemp=new DenseDoubleMatrix2D(_opt.hiddenStateSize+20,_opt.hiddenStateSize+20);
		DenseDoubleMatrix2D UhatTemp1=new DenseDoubleMatrix2D(_opt.hiddenStateSize+20,_opt.hiddenStateSize+20);
		
		DenseDoubleMatrix2D Uhat=new DenseDoubleMatrix2D(_opt.hiddenStateSize+20,_opt.hiddenStateSize);
		
		SparseDoubleMatrix2D sValsOmega=new SparseDoubleMatrix2D(_opt.hiddenStateSize+20,_opt.hiddenStateSize+20);
		DenseDoubleMatrix2D b=new DenseDoubleMatrix2D(_opt.hiddenStateSize+20,_dim2);
		DenseDoubleMatrix2D q=new DenseDoubleMatrix2D(X.rows(),_opt.hiddenStateSize+20);

		System.out.println("====Starting Power Iteration====");
		for(int powIter =0; powIter <5;powIter++){
			startTime = System.currentTimeMillis();
			X.zMult(omega, Xomega);
			endTime = System.currentTimeMillis();
		System.out.println("===Time taken for Multiplication: "+(endTime-startTime)/1000.0/60.0+" mins===");

			startTime = System.currentTimeMillis();

			QR qr=new QR(Xomega.rows(),Xomega.columns());


			DenseMatrix XomegaMTJ=MatrixFormatConversion.createDenseMatrixMTJ(Xomega);
			DenseMatrix qMTJ=qr.factor(XomegaMTJ).getQ();
			q=MatrixFormatConversion.createDenseMatrixCOLT(qMTJ);

			DenseDoubleAlgebra dalg=new DenseDoubleAlgebra();
			DenseDoubleMatrix2D qt=(DenseDoubleMatrix2D) dalg.transpose(q);

			qt.zMult(X, b);
			omega= (DenseDoubleMatrix2D)dalg.transpose(b);
			endTime = System.currentTimeMillis();
		}
		System.out.println("===Time taken for QR and multiply: "+(endTime-startTime)/1000.0/60.0+" mins===");

		startTime = System.currentTimeMillis();

		DenseDoubleSingularValueDecomposition svd= new DenseDoubleSingularValueDecomposition(b,true,false);
		
		
		UhatTemp1=(DenseDoubleMatrix2D) svd.getU();
		
		setSingularVals(svd.getSingularValues());
		
		if(_opt.scaleBySingVals){
			sValsOmega=(SparseDoubleMatrix2D) svd.getS();
			sValsOmega.normalize();
			UhatTemp1.zMult(sValsOmega, UhatTemp);
		}
		else{
			UhatTemp=UhatTemp1;
		}
		
		
		for(int i=0; i<_opt.hiddenStateSize+20;i++){ //Take only the top k elements of the matrix after svd.
			for (int j=0;j<_opt.hiddenStateSize;j++){
				Uhat.set(i, j, UhatTemp.get(i, j));
			}
		}
		
		endTime = System.currentTimeMillis();
		System.out.println("===Time taken for SVD: "+(endTime-startTime)/1000.0/60.0+" mins===");




		startTime = System.currentTimeMillis();
		q.zMult(Uhat, dictMatrixCOLT);
		endTime = System.currentTimeMillis();
		System.out.println("===Time taken for Final Multiply: "+(endTime-startTime)/1000.0/60.0+" mins===");

		return MatrixFormatConversion.createDenseMatrixJAMA(dictMatrixCOLT);
	
}
	
	
	public Matrix computeSVD_Tropp(DenseDoubleMatrix2D X, DenseDoubleMatrix2D omega, int _dim2)
	{
		dictMatrixCOLT=new DenseDoubleMatrix2D(X.rows(),_opt.hiddenStateSize);
		DenseDoubleMatrix2D Xomega=new DenseDoubleMatrix2D(X.rows(),_opt.hiddenStateSize+20);//Oversample the required rank.
		DenseDoubleMatrix2D UhatTemp=new DenseDoubleMatrix2D(_opt.hiddenStateSize+20,_opt.hiddenStateSize+20);
		DenseDoubleMatrix2D UhatTemp1=new DenseDoubleMatrix2D(_opt.hiddenStateSize+20,_opt.hiddenStateSize+20);
		
		DenseDoubleMatrix2D Uhat=new DenseDoubleMatrix2D(_opt.hiddenStateSize+20,_opt.hiddenStateSize);
		
		SparseDoubleMatrix2D sValsOmega=new SparseDoubleMatrix2D(_opt.hiddenStateSize+20,_opt.hiddenStateSize+20);
		DenseDoubleMatrix2D b=new DenseDoubleMatrix2D(_opt.hiddenStateSize+20,_dim2);
		DenseDoubleMatrix2D q=new DenseDoubleMatrix2D(X.rows(),_opt.hiddenStateSize+20);

		System.out.println("====Starting Power Iteration====");
		for(int powIter =0; powIter <5;powIter++){
			startTime = System.currentTimeMillis();
			X.zMult(omega, Xomega);
			endTime = System.currentTimeMillis();
		System.out.println("===Time taken for Multiplication: "+(endTime-startTime)/1000.0/60.0+" mins===");

			startTime = System.currentTimeMillis();

			QR qr=new QR(Xomega.rows(),Xomega.columns());


			DenseMatrix XomegaMTJ=MatrixFormatConversion.createDenseMatrixMTJ(Xomega);
			DenseMatrix qMTJ=qr.factor(XomegaMTJ).getQ();
			q=MatrixFormatConversion.createDenseMatrixCOLT(qMTJ);

			DenseDoubleAlgebra dalg=new DenseDoubleAlgebra();
			DenseDoubleMatrix2D qt=(DenseDoubleMatrix2D) dalg.transpose(q);

			qt.zMult(X, b);
			omega= (DenseDoubleMatrix2D)dalg.transpose(b);
			endTime = System.currentTimeMillis();
		}
		System.out.println("===Time taken for QR and multiply: "+(endTime-startTime)/1000.0/60.0+" mins===");

		startTime = System.currentTimeMillis();

		DenseDoubleSingularValueDecomposition svd= new DenseDoubleSingularValueDecomposition(b,true,false);
		
		
		UhatTemp1=(DenseDoubleMatrix2D) svd.getU();
		
		setSingularVals(svd.getSingularValues());
		
		if(_opt.scaleBySingVals){
			sValsOmega=(SparseDoubleMatrix2D) svd.getS();
			sValsOmega.normalize();
			UhatTemp1.zMult(sValsOmega, UhatTemp);
		}
		else{
			UhatTemp=UhatTemp1;
		}
		
		
		for(int i=0; i<_opt.hiddenStateSize+20;i++){ //Take only the top k elements of the matrix after svd.
			for (int j=0;j<_opt.hiddenStateSize;j++){
				Uhat.set(i, j, UhatTemp.get(i, j));
			}
		}
		
		endTime = System.currentTimeMillis();
		System.out.println("===Time taken for SVD: "+(endTime-startTime)/1000.0/60.0+" mins===");




		startTime = System.currentTimeMillis();
		q.zMult(Uhat, dictMatrixCOLT);
		endTime = System.currentTimeMillis();
		System.out.println("===Time taken for Final Multiply: "+(endTime-startTime)/1000.0/60.0+" mins===");

		return MatrixFormatConversion.createDenseMatrixJAMA(dictMatrixCOLT);
	
}
	

	private void setSingularVals(double[] singularValues) {
		this.sVals=singularValues;
		
	}

	public double[] getSingularVals(){
		return sVals;
	}


}
