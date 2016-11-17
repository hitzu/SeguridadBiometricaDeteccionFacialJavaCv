/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Escritorio;

import com.atul.JavaOpenCV.Imshow;
import java.awt.Graphics;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.ByteArrayInputStream;
import javax.imageio.ImageIO;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.highgui.Highgui;
import org.opencv.highgui.VideoCapture;
import org.opencv.objdetect.CascadeClassifier;
import static org.opencv.objdetect.Objdetect.CASCADE_SCALE_IMAGE;
import java.util.LinkedList;
import java.io.File;
import java.nio.IntBuffer;
import javax.swing.JOptionPane;
import org.bytedeco.javacpp.Loader;
import static org.bytedeco.javacpp.opencv_core.CV_32SC1;
import org.bytedeco.javacpp.opencv_core.MatVector;
import org.bytedeco.javacpp.opencv_face;

import static org.bytedeco.javacpp.opencv_face.createFisherFaceRecognizer;
import org.bytedeco.javacpp.opencv_objdetect;
import org.bytedeco.javacv.Java2DFrameConverter;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.opencv.imgproc.Imgproc;


/**
 *
 * @author Hitzu
 */
public class Interfaz extends javax.swing.JFrame {
    
    private HiloDemonio miHilo = null;
    private HiloAdministrador miHilo1 = null;
    private HiloCaptura miHilo2 = null;
    int count = 0;
    VideoCapture video = null;
    Mat frame = new Mat();
    MatOfByte mem = new MatOfByte();
    CascadeClassifier detectorRostros = new CascadeClassifier("/home/hitzu/opencv-2.4.13/data/haarcascades/haarcascade_frontalface_alt2.xml");
    //CascadeClassifier detectorRostros = new CascadeClassifier("haarcascade_frontalface_alt2.xml");
    MatOfRect rostros = new MatOfRect();
    Rect[] facesArray;
    //Lista ligada de los elementos
    LinkedList<Mat> imagenes = new LinkedList<>();
    LinkedList<String> etiquetas = new LinkedList<>();
    LinkedList<String> nombres = new LinkedList<>();
    MatVector images;
    org.bytedeco.javacpp.opencv_core.Mat labels;
    opencv_face.FaceRecognizer faceRecognizer;

    public Interfaz() 
    {
        
        cargar_modelos();
        initComponents();
        Captura.setEnabled(false);
        
    }
    
    public void cargar_modelos()
    {
        //Aqui se recorren los directorios para buscar las imagenes de las distintas clases
        String directorio = "/home/hitzu/NetBeansProjects/ReconocimientoFacialJavaCv/dist/clases";
        File f = new File(directorio);
        Mat aux,faceROI;
        if(f.exists())
        {
            File[] ficheros = f.listFiles();
            for(File fichero : ficheros)
            {
                String[] separador = fichero.getName().split(",");
                aux = Highgui.imread(fichero.getAbsolutePath(), Highgui.CV_LOAD_IMAGE_COLOR);
                if(isFace(aux))
                {
                    aux = DetectFace(aux);
                    aux = Convert2Gray(aux);
                    faceROI = Equalize(aux);
                    Size sz = new Size(200,200);
                    Imgproc.resize(faceROI, aux, sz);
                    FillVectors(aux,""+separador[1].charAt(0),separador[0].substring(0, (separador[0].length()-2)));
                }
            }
        }
        //mostrar();
        convertMatToMat();
        convertArrayIntToOpencvMat();
        CreateModel();
    }
    
    public void mostrar()
    {
        for(String etiqueta : etiquetas)
        {
            System.out.println(etiqueta);
        }
        //Imshow image = new Imshow("Gris");
        //image.showImage(imagenes.getFirst());
    }
    
    public boolean isFace(Mat imagen)
    {
        //cargando el clasificador en cascada para la deteccion de rostros
        Rect[] facesArray;
        MatOfRect rostros = new MatOfRect();
        
        detectorRostros.detectMultiScale(imagen, rostros, 1.1, 2, 0|CASCADE_SCALE_IMAGE, new Size(30, 30), new Size(imagen.height(), imagen.width() ) );
        facesArray = rostros.toArray();
        return facesArray.length >= 1;
    }
    
    public Mat DetectFace(Mat aux)
    {        
        Rect[] facesArray;
        MatOfRect rostros = new MatOfRect();
        Mat faceROI;
        
        detectorRostros.detectMultiScale(aux, rostros, 1.1, 2, 0|CASCADE_SCALE_IMAGE, new Size(30, 30), new Size(aux.height(), aux.width() ) );
        facesArray = rostros.toArray();

        faceROI = aux.submat(facesArray[0]);

        return faceROI;
    }
    
    
    public Mat Convert2Gray(Mat imagen)
    {
        Mat aux = new Mat();
        Imgproc.cvtColor(imagen, aux, Imgproc.COLOR_BGR2GRAY);
        return aux;
    }
    
    public Mat Equalize(Mat imagen)
    {
        Mat aux = new Mat();
        Imgproc.equalizeHist(imagen, aux);
        return aux;
    }
    
    public void FillVectors(Mat imagen,String clase,String nombre)
    {
        imagenes.add(imagen);
        etiquetas.add(clase);
        nombres.add(nombre);
    }
    
    public void convertMatToMat()
    {
        BufferedImage image;
        int type = BufferedImage.TYPE_BYTE_GRAY;
        images = new MatVector(imagenes.size()); 
        
        long cont = 0;
        for(int i = 0; i< imagenes.size(); i++)
        {
            //primera conversion
            if(imagenes.get(i).channels() > 1)
            {
                type = BufferedImage.TYPE_3BYTE_BGR;
            }
            image = new BufferedImage(imagenes.get(i).cols(), imagenes.get(i).rows(), type);
            imagenes.get(i).get(0,0, ((DataBufferByte)image.getRaster().getDataBuffer()).getData());
            //segunda conversion
            OpenCVFrameConverter.ToMat cv = new OpenCVFrameConverter.ToMat(); 
            org.bytedeco.javacpp.opencv_core.Mat resultado = cv.convertToMat(new Java2DFrameConverter().convert(image));
            images.put(cont,resultado);
            cont++;
        }
    }
    
    public void convertArrayIntToOpencvMat()
    {
        labels = new org.bytedeco.javacpp.opencv_core.Mat(etiquetas.size(),1,CV_32SC1);
        IntBuffer labelsBuf = labels.createBuffer();
        for(int i = 0; i < etiquetas.size(); i++)
        {
            labelsBuf.put(i,(Integer.parseInt(etiquetas.get(i))));
        }   
    }
    
    public void CreateModel()
    {
        faceRecognizer = createFisherFaceRecognizer();
        faceRecognizer.train(images, labels);
    }
    
    public int detectar(Mat aux)
    {
        if(isFace(aux))
        {
            aux = DetectFace(aux);
            aux = Convert2Gray(aux);
            aux = Equalize(aux);
            Size sz = new Size(200,200);
            Imgproc.resize(aux, aux, sz);
        }
        
        //algoritmo para convertir una sola imagen
        BufferedImage image;
        int type = BufferedImage.TYPE_BYTE_GRAY;

        if(aux.channels() > 1)
        {
            type = BufferedImage.TYPE_3BYTE_BGR;
        }
        image = new BufferedImage(aux.cols(), aux.rows(), type);
        aux.get(0,0, ((DataBufferByte)image.getRaster().getDataBuffer()).getData());
        //segunda conversion
        OpenCVFrameConverter.ToMat cv = new OpenCVFrameConverter.ToMat(); 
        org.bytedeco.javacpp.opencv_core.Mat resultado = cv.convertToMat(new Java2DFrameConverter().convert(image));
        
        int predictedLabel = faceRecognizer.predict(resultado);
        return predictedLabel;
    }
    
    public void EsAdministrador()
    {
        JOptionPane.showMessageDialog(null, "Eres administrador puedes agregar a un usuario");
        Captura.setEnabled(true);
        pausa1();
        imagen.removeAll();
        imagen.repaint();
        AgregarUsuario();
    }
    
    public void AgregarUsuario()
    {            
        //prender la camara con hilo captura
        video = new VideoCapture(0);
        miHilo2 = new HiloCaptura();
        Thread t = new Thread(miHilo2);
        t.setDaemon(true);
        miHilo2.nombre = JOptionPane.showInputDialog(null, "Ingresa el primer nombre de la persona a registrar");
        miHilo2.apellido = JOptionPane.showInputDialog(null, "Ingresa el primer apellido de la persona a registrar");
        JOptionPane.showMessageDialog(null, "Se deben de tomar 5 fotos de la persona a registrar");
        miHilo2.runnable = true;
        t.start();
        
    }
    
    public int numero_clases()
    {
        int mayor = 0;
        for(String etiqueta:etiquetas)
        {
            if(Integer.parseInt(etiqueta)>mayor)
                mayor = Integer.parseInt(etiqueta);
        }
        
        return mayor;
    }
    
    public boolean GuardarImagen(Mat imagen, String nombre, String apellido, int contador)
    {
        //si el reconocimiento es bueno
        detectorRostros.detectMultiScale(imagen, rostros, 1.1, 2, 0|CASCADE_SCALE_IMAGE, new Size(80, 80), new Size(frame.height(), frame.width() ) );
        facesArray = rostros.toArray();
        if(facesArray.length == 1)
        {
            int clases = numero_clases() + 1;
            Highgui.imwrite("/home/hitzu/NetBeansProjects/ReconocimientoFacialJavaCv/dist/clases/"+nombre+"_"+apellido+"0"+contador+","+clases+".jpg", imagen);
            return true;
        }
        return false;
    }
    
    class HiloCaptura implements Runnable
    {
        protected volatile boolean runnable = false;
        protected volatile String nombre = null;
        protected volatile String apellido = null;
        int contador = 1;
        @Override
        public void run()
        {
            synchronized(this)
            {
                while(runnable)
                {
                    if(video.grab())
                    {
                        try
                        {
                            video.retrieve(frame);
                            
                            Highgui.imencode(".bmp",frame,mem);
                            Image im = ImageIO.read(new ByteArrayInputStream(mem.toArray()));
                            
                            BufferedImage buff = (BufferedImage) im;
                            Graphics gr = imagen.getGraphics();
                            
                            if(gr.drawImage(buff, 0, 0, getWidth(), getHeight() -150 , 0, 0, buff.getWidth(), buff.getHeight(), null))
                                
                            if(runnable == false)
                            {
                                if(GuardarImagen(frame, nombre, apellido, contador))
                                {
                                    JOptionPane.showMessageDialog(null, "Se ha ingresado la imagen: " + contador);
                                    contador++;
                                }
                                else
                                {
                                    JOptionPane.showMessageDialog(null, "no se ha detectado un rostro en la imagen tomada");
                                }
                                runnable = true;
                            }
                        }
                        catch(Exception e)
                        {
                            System.out.println("error: " + e.toString());
                        }
                    }   
                }
            }
        }
        
    }
    
    class HiloAdministrador implements Runnable
    {
        protected volatile boolean runnable = false;
        int respuesta = -1;
        
        @Override
        public void run()
        {
            synchronized(this)
            {
                while(runnable)
                {
                    if(video.grab())
                    {
                        try
                        {
                            video.retrieve(frame);
                            /*Empieza el desmadre para detectar rostros*/
                            respuesta = detectar(frame);
                            
                            detectorRostros.detectMultiScale(frame, rostros, 1.1, 2, 0|CASCADE_SCALE_IMAGE, new Size(80, 80), new Size(frame.height(), frame.width() ) );
                            facesArray = rostros.toArray();
                            
                            for(int i = 0; i < facesArray.length; i++)
                            {
                                Core.rectangle(frame,
                                new Point(facesArray[i].x,facesArray[i].y),
                                new Point(facesArray[i].x+facesArray[i].width,facesArray[i].y+facesArray[i].height),
                                new Scalar(255,0,0));
                            }
                            //if(respuesta == 3)
                            if(true)
                            {
                                EsAdministrador();
                            }
                            /*termina el desmadre para detectar rostros*/
                            
                            Highgui.imencode(".bmp",frame,mem);
                            Image im = ImageIO.read(new ByteArrayInputStream(mem.toArray()));
                            
                            BufferedImage buff = (BufferedImage) im;
                            Graphics gr = imagen.getGraphics();
                            
                            if(gr.drawImage(buff, 0, 0, getWidth(), getHeight() -150 , 0, 0, buff.getWidth(), buff.getHeight(), null))
                                
                            if(runnable == false)
                            {
                                System.out.println("La camara de administrador esta en pausa");
                                this.wait();
                            }
                        }
                        catch(Exception e)
                        {
                            System.out.println("error: " + e.toString());
                        }
                    }   
                }
            }
        }
        
    }
    
    
    class HiloDemonio implements Runnable
    {
        protected volatile boolean runnable = false;
        int respuesta = -1;
        
        @Override
        public void run()
        {
            synchronized(this)
            {
                while(runnable)
                {
                    if(video.grab())
                    {
                        try
                        {
                            video.retrieve(frame);
                            /*Empieza el desmadre para detectar rostros*/
                            respuesta = detectar(frame);
                            
                            detectorRostros.detectMultiScale(frame, rostros, 1.1, 2, 0|CASCADE_SCALE_IMAGE, new Size(80, 80), new Size(frame.height(), frame.width() ) );
                            facesArray = rostros.toArray();                            
                            for(int i = 0; i < facesArray.length; i++)
                            {
                                if( (respuesta  == 1) || (respuesta ==2))
                                {
                                    Core.rectangle(frame,new Point(facesArray[i].x,facesArray[i].y),new Point(facesArray[i].x+facesArray[i].width,facesArray[i].y+facesArray[i].height),
                                    new Scalar(255,0,0));
                                    System.out.println("usuario no registrado");
                                }
                                else if(respuesta == 3)
                                {
                                    Core.rectangle(frame,new Point(facesArray[i].x,facesArray[i].y),new Point(facesArray[i].x+facesArray[i].width,facesArray[i].y+facesArray[i].height),
                                    new Scalar(0,255,0));
                                    System.out.println("usuario administrador: " + nombres.get(etiquetas.indexOf(String.valueOf(respuesta))));
                                }
                                else
                                {
                                    Core.rectangle(frame,new Point(facesArray[i].x,facesArray[i].y),new Point(facesArray[i].x+facesArray[i].width,facesArray[i].y+facesArray[i].height),
                                    new Scalar(0,0,255));
                                    System.out.println("usuario comun: " + nombres.get(etiquetas.indexOf(String.valueOf(respuesta))));
                                }      
                                
                            }
                             
                            /*termina el desmadre para detectar rostros*/
                            
                            Highgui.imencode(".bmp",frame,mem);
                            Image im = ImageIO.read(new ByteArrayInputStream(mem.toArray()));
                            
                            BufferedImage buff = (BufferedImage) im;
                            Graphics gr = imagen.getGraphics();
                            
                            if(gr.drawImage(buff, 0, 0, getWidth(), getHeight() -150 , 0, 0, buff.getWidth(), buff.getHeight(), null))
                                
                            if(runnable == false)
                            {
                                System.out.println("La camara esta en pausa");
                                this.wait();
                            }
                        }
                        catch(Exception e)
                        {
                            System.out.println("error: " + e.toString());
                        }
                    }   
                }
            }
        }
    }

    /**
     * This method is called from within the constructor to initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is always
     * regenerated by the Form Editor.
     */
    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        jPanel1 = new javax.swing.JPanel();
        imagen = new javax.swing.JPanel();
        start = new javax.swing.JButton();
        stop = new javax.swing.JButton();
        Eliminar = new javax.swing.JButton();
        Agrega = new javax.swing.JButton();
        Captura = new javax.swing.JButton();

        setDefaultCloseOperation(javax.swing.WindowConstants.EXIT_ON_CLOSE);
        setBackground(new java.awt.Color(54, 54, 54));
        setSize(new java.awt.Dimension(1360, 768));

        jPanel1.setBackground(new java.awt.Color(22, 11, 0));
        jPanel1.setPreferredSize(new java.awt.Dimension(1200, 800));

        imagen.setPreferredSize(new java.awt.Dimension(900, 900));
        imagen.setVerifyInputWhenFocusTarget(false);

        javax.swing.GroupLayout imagenLayout = new javax.swing.GroupLayout(imagen);
        imagen.setLayout(imagenLayout);
        imagenLayout.setHorizontalGroup(
            imagenLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 900, Short.MAX_VALUE)
        );
        imagenLayout.setVerticalGroup(
            imagenLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 551, Short.MAX_VALUE)
        );

        start.setText("Iniciar reconocimiento");
        start.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                startActionPerformed(evt);
            }
        });

        stop.setText("Detener reconocimiento");
        stop.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                stopActionPerformed(evt);
            }
        });

        Eliminar.setText("Eliminar usuario");

        Agrega.setText("Agregar usuario");
        Agrega.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                AgregaActionPerformed(evt);
            }
        });

        Captura.setText("Tomar Foto");
        Captura.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                CapturaActionPerformed(evt);
            }
        });

        javax.swing.GroupLayout jPanel1Layout = new javax.swing.GroupLayout(jPanel1);
        jPanel1.setLayout(jPanel1Layout);
        jPanel1Layout.setHorizontalGroup(
            jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(jPanel1Layout.createSequentialGroup()
                .addGroup(jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addGroup(jPanel1Layout.createSequentialGroup()
                        .addGap(444, 444, 444)
                        .addComponent(Captura, javax.swing.GroupLayout.PREFERRED_SIZE, 263, javax.swing.GroupLayout.PREFERRED_SIZE))
                    .addGroup(jPanel1Layout.createSequentialGroup()
                        .addGap(117, 117, 117)
                        .addComponent(imagen, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)))
                .addGap(97, 97, 97)
                .addGroup(jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(Eliminar, javax.swing.GroupLayout.PREFERRED_SIZE, 193, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(Agrega, javax.swing.GroupLayout.PREFERRED_SIZE, 193, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(start, javax.swing.GroupLayout.PREFERRED_SIZE, 193, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(stop, javax.swing.GroupLayout.PREFERRED_SIZE, 193, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addContainerGap(43, Short.MAX_VALUE))
        );
        jPanel1Layout.setVerticalGroup(
            jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(jPanel1Layout.createSequentialGroup()
                .addGap(28, 28, 28)
                .addGroup(jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addGroup(jPanel1Layout.createSequentialGroup()
                        .addComponent(Agrega, javax.swing.GroupLayout.PREFERRED_SIZE, 135, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addGap(47, 47, 47)
                        .addComponent(Eliminar, javax.swing.GroupLayout.PREFERRED_SIZE, 135, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addGap(52, 52, 52)
                        .addComponent(start, javax.swing.GroupLayout.PREFERRED_SIZE, 134, javax.swing.GroupLayout.PREFERRED_SIZE))
                    .addComponent(imagen, javax.swing.GroupLayout.PREFERRED_SIZE, 551, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                .addGroup(jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(stop, javax.swing.GroupLayout.PREFERRED_SIZE, 131, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(Captura, javax.swing.GroupLayout.PREFERRED_SIZE, 131, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addGap(0, 38, Short.MAX_VALUE))
        );

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(getContentPane());
        getContentPane().setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addComponent(jPanel1, javax.swing.GroupLayout.PREFERRED_SIZE, 1350, javax.swing.GroupLayout.PREFERRED_SIZE)
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addComponent(jPanel1, javax.swing.GroupLayout.PREFERRED_SIZE, 760, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addGap(0, 40, Short.MAX_VALUE))
        );

        pack();
    }// </editor-fold>//GEN-END:initComponents

    private void AgregaActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_AgregaActionPerformed
        // TODO add your handling code here:
        video = new VideoCapture(0);
        miHilo1 = new HiloAdministrador();
        Thread t = new Thread(miHilo1);
        t.setDaemon(true);
        miHilo1.runnable = true;
        t.start();
    }//GEN-LAST:event_AgregaActionPerformed

    private void stopActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_stopActionPerformed
        // TODO add your handling code here:
        pausa();
    }//GEN-LAST:event_stopActionPerformed

    private void startActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_startActionPerformed
        // TODO add your handling code here:
        video = new VideoCapture(0);
        miHilo = new HiloDemonio();
        Thread t = new Thread(miHilo);
        t.setDaemon(true);
        miHilo.runnable = true;
        t.start();
        start.setEnabled(false);
        stop.setEnabled(true);

    }//GEN-LAST:event_startActionPerformed

    private void CapturaActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_CapturaActionPerformed
        // TODO add your handling code here:
        miHilo2.runnable = false;
    }//GEN-LAST:event_CapturaActionPerformed

    
    public void pausa()
    {
        miHilo.runnable = false;
        stop.setEnabled(false);
        start.setEnabled(true);
        imagen.removeAll();
        imagen.repaint();
        video.release();
    }
    
    public void pausa1()
    {
        miHilo1.runnable = false;
        video.release();
    }
    
    /**
     * @param args the command line arguments
     */
    public static void main(String args[]) {
        /* Set the Nimbus look and feel */
        //<editor-fold defaultstate="collapsed" desc=" Look and feel setting code (optional) ">
        /* If Nimbus (introduced in Java SE 6) is not available, stay with the default look and feel.
         * For details see http://download.oracle.com/javase/tutorial/uiswing/lookandfeel/plaf.html 
         */
        try {
            for (javax.swing.UIManager.LookAndFeelInfo info : javax.swing.UIManager.getInstalledLookAndFeels()) {
                if ("Nimbus".equals(info.getName())) {
                    javax.swing.UIManager.setLookAndFeel(info.getClassName());
                    break;
                }
            }
        } catch (ClassNotFoundException ex) {
            java.util.logging.Logger.getLogger(Interfaz.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (InstantiationException ex) {
            java.util.logging.Logger.getLogger(Interfaz.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (IllegalAccessException ex) {
            java.util.logging.Logger.getLogger(Interfaz.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (javax.swing.UnsupportedLookAndFeelException ex) {
            java.util.logging.Logger.getLogger(Interfaz.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        }
        //</editor-fold>

        Loader.load(opencv_objdetect.class);
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        /* Create and display the form */
        java.awt.EventQueue.invokeLater(new Runnable() {
            public void run() {
                new Interfaz().setVisible(true);
            }
        });
    }

    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JButton Agrega;
    private javax.swing.JButton Captura;
    private javax.swing.JButton Eliminar;
    private javax.swing.JPanel imagen;
    private javax.swing.JPanel jPanel1;
    private javax.swing.JButton start;
    private javax.swing.JButton stop;
    // End of variables declaration//GEN-END:variables
}
