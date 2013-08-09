package edu.upenn.cis.SpectralLearning.MathUtils;

import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.sparse.CompColMatrix;
import Jama.Matrix;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix2D;

public class MatrixFormatConversion {
	
	
	
	public static Matrix createDenseMatrixJAMA(DenseDoubleMatrix2D xCOLT) {
		Matrix x=new Matrix(xCOLT.rows(),xCOLT.columns());
		for (int i=0;i<xCOLT.rows();i++){
			for (int j=0;j<xCOLT.columns();j++){
				x.set(i, j, xCOLT.get(i, j));
			}
		}
	
		return x;
	}
	
	public static DenseMatrix createDenseMatrixMTJ(DenseDoubleMatrix2D xomega) {
		DenseMatrix xMTJ=new DenseMatrix(xomega.rows(),xomega.columns());
				
		
		for (int i=0;i<xomega.rows();i++){
			for (int j=0;j<xomega.columns();j++){
				xMTJ.set(i, j, xomega.getQuick(i, j));
			}
		}
		return xMTJ;
	}

	public static DenseDoubleMatrix2D createDenseMatrixCOLT(DenseMatrix xmtj) {
		DenseDoubleMatrix2D x_omega=new DenseDoubleMatrix2D(xmtj.numRows(),xmtj.numColumns());
		for (int i=0;i<xmtj.numRows();i++){
			for (int j=0;j<xmtj.numColumns();j++){
				
				x_omega.set(i, j, xmtj.get(i, j));
			}
		}
		return x_omega;
	}

	public static DenseDoubleMatrix2D createDenseMatrixCOLT(Matrix xJama) {
		DenseDoubleMatrix2D x_omega=new DenseDoubleMatrix2D(xJama.getRowDimension(),xJama.getColumnDimension());
		for (int i=0;i<xJama.getRowDimension();i++){
			for (int j=0;j<xJama.getColumnDimension();j++){
				x_omega.set(i, j, xJama.get(i, j));
			}
		}
		return x_omega;
	}
	

}
