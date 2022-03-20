/*
 *	Calcula la Entropia
 */
import java.io.*;

class Entropia{
	static RandomAccessFile Datos;
	static PrintStream p=System.out;
	static BufferedReader Kbr=new BufferedReader(new InputStreamReader(System.in));

    public static void main(String[] args) throws Exception {
	  String sResp;
	  int i;
   	  for (int loop=0;loop<10000;loop++){
   	  	if (loop!=0){
		    p.println("\nDesea leer otro archivo?");
		    p.println("\"S\" para continuar; otro para terminar...)");
			sResp=Kbr.readLine().toUpperCase();
	   		if (!sResp.equals("S"))
				return;
			//endIf
		}//endif
		p.println("Deme el nombre del archivo de datos cuya entropia quiere calcular:");
		//String FName=Kbr.readLine().toUpperCase();
		String FName=Kbr.readLine();
	  	try {
	  		Datos=new RandomAccessFile(new File(FName), "r");
	  		p.println();
	  	}//endTry
  		catch (Exception e1){
  			p.println("No se encontro \""+FName+"\"");
			continue;
  		}//endCatch
  		PrintStream Fps;
  		while (true){
		  p.println("Deme el nombre del archivo para almacenar los resultados:");
		  String Res=Kbr.readLine();
  	      try{Fps=new PrintStream(new FileOutputStream(new File(Res)));}
  	      catch (Exception e2) {
  	    	p.println("Error al crear el archivo de salida");
  	    	continue;
  	      }//endCatch
  	      break;
  	    }//endWhile
		int BytesEnDatos=0;
		byte X;
		while (true){
			Datos.seek(BytesEnDatos);
			try{X=Datos.readByte();}
			catch (Exception e){break;}
			BytesEnDatos++;
		}//endWhile
		Datos.close();
		p.println("Se leyeron "+BytesEnDatos+" bytes\n");
		p.println("\nInicio calculo de la Entropia...\n");
		Double dBytesEnDatos=(double)BytesEnDatos;
  		Datos=new RandomAccessFile(new File(FName), "r");
		int Veces[]=new int [256];
		for (i=0;i<256;i++)
			Veces[i]=0;
		int Index;
		for (i=0;i<BytesEnDatos;i++){
			Datos.seek(i);
			Index=Datos.read();
			Veces[Index]++;
		}//endFor
		Datos.close();
		Double Prob[]=new Double [256];
		for (i=0;i<256;i++){
			Prob[i]=(double)Veces[i]/dBytesEnDatos;
		}//endFor
		Double Entropy=0d;
		Double log2=Math.log(2.0d);
		for (i=0;i<256;i++){
			p.printf("Prob[%3.0f]= %12.10f \n",(float)i,Prob[i]);
			Fps.printf("Prob[%3.0f]= %12.10f",(float)i,Prob[i]);
			Fps.println();
			if (Prob[i]!=0){
				Entropy=Entropy-Prob[i]*Math.log(Prob[i])/log2;
			}//endif
		}//endFor
		p.println("La entropia calculada es "+Entropy);
		Fps.println("La entropia calculada es "+Entropy);
	  }//endLoop
   }//endMain
}//endClass

