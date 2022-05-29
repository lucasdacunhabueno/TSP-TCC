package GA;

import java.io.BufferedReader;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;

import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import org.jocl.CL;
import static org.jocl.CL.clGetPlatformIDs;
import org.jocl.Pointer;
import org.jocl.cl_platform_id;
import static org.jocl.CL.*;
import org.jocl.*;

public class GA_JOCL {

   private static String programSource
           = "__kernel void "
           + "sampleKernel(__global const float *x1,"
           + "             __global const float *x2,"
           + "             __global const float *y1,"
           + "             __global const float *y2,"
           + "             __global float *c)"
           + "{"
           + "    int gid = get_global_id(0);"
           + "    c[gid]  = sqrt((float)((((x1[gid] - x2[gid]) * (x1[gid] - x2[gid])) + ((y1[gid] - y2[gid]) * (y1[gid] - y2[gid]))) / 10));"
           + "}";

   private Chromosome[] chromosomes;
   private Chromosome[] nextGeneration;
   private int N;
   private int cityNum;
   private double p_c_t;
   private double p_m_t;
   private int MAX_GEN;
   private int bestLength;
   private int[] bestTour;
   private double bestFitness;
   private double[] averageFitness;
   private int[][] distance;
   private String filename;

   public GA_JOCL() {
      N = 100;
      cityNum = 30;
      p_c_t = 0.9;
      p_m_t = 0.1;
      MAX_GEN = 1000;
      bestLength = 0;
      bestTour = new int[cityNum];
      bestFitness = 0.0;
      averageFitness = new double[MAX_GEN];
      chromosomes = new Chromosome[N];
      distance = new int[cityNum][cityNum];

   }

   /**
    * Constructor of GA class
    *
    * @param n TAMANHO DA POPULAÇÃO
    * @param num NÚMERO DE CIDADES
    * @param g NÚMERO DE GERAÇÕES
    * @param p_c TAXA DE CROSSOVER
    * @param p_m TAXA DE MUTAÇÃO
    * @param filename NOME DO ARQUIO
    */
   public GA_JOCL(int n, int num, int g, double p_c, double p_m, String filename) {
      this.N = n;
      this.cityNum = num;
      this.MAX_GEN = g;
      this.p_c_t = p_c;
      this.p_m_t = p_m;
      bestTour = new int[cityNum];
      averageFitness = new double[MAX_GEN];
      bestFitness = 0.0;
      chromosomes = new Chromosome[N];
      nextGeneration = new Chromosome[N];
      distance = new int[cityNum][cityNum];
      this.filename = filename;
   }

   public void solve() throws IOException {
//        System.out.println("---------------------Start initilization---------------------");
      init();
//        System.out.println("---------------------End initilization---------------------");
//        System.out.println("---------------------Start evolution---------------------");
      for (int i = 0; i < MAX_GEN; i++) {
         System.gc();
         System.runFinalization();
//            System.out.println("-----------Start generation " + i + "----------");
         evolve(i);
//            System.out.println("-----------End generation " + i + "----------");
      }
//        System.out.println("---------------------End evolution---------------------");
      printOptimal();
//        System.exit(1);
//        outputResults();

   }

   /**
    * ��ʼ��GA
    *
    * @throws IOException
    */
   @SuppressWarnings("resource")
   private void init() throws IOException {
      int[] x;
      int[] y;
      String strbuff;
      BufferedReader data = new BufferedReader(new InputStreamReader(new FileInputStream(filename)));

      distance = new int[cityNum][cityNum];
      x = new int[cityNum];
      y = new int[cityNum];
      while ((strbuff = data.readLine()) != null) {
         if (!Character.isAlphabetic(strbuff.charAt(0))) {
            break;
         }
      }
      String[] tmp = strbuff.split(" ");
      x[0] = Integer.valueOf(tmp[1]);
      y[0] = Integer.valueOf(tmp[2]);
      for (int i = 1; i < cityNum; i++) {
         strbuff = data.readLine();
         String[] strcol = strbuff.split(" ");
         x[i] = Integer.valueOf(strcol[1]);
         y[i] = Integer.valueOf(strcol[2]);
      }

      ArrayList<CityPair> listCities = new ArrayList<>();

      for (int i = 0; i < cityNum - 1; i++) {
         listCities.add(new CityPair(0, 0, 0, 0));
         for (int j = i + 1; j < cityNum; j++) {
            listCities.add(new CityPair(x[i], x[j], y[i], y[j]));
         }
      }
      listCities.add(new CityPair(0, 0, 0, 0));

      float[] response = getDistanceBetweenCities(listCities);
      int iterador = 0;
      for (int i = 0; i < cityNum - 1; i++) {
         distance[i][i] = (int) response[iterador];
         iterador++;
         for (int j = i + 1; j < cityNum; j++) {
            distance[i][j] = (int) response[iterador];
            distance[j][i] = (int) response[iterador];
            iterador++;
         }
      }
      distance[cityNum - 1][cityNum - 1] = 0;

      for (int i = 0; i < N; i++) {
         Chromosome chromosome = new Chromosome(cityNum, distance);
         chromosome.randomGeneration();
         chromosomes[i] = chromosome;
      }
   }

   private float[] getDistanceBetweenCities(ArrayList<CityPair> listCities) {

      CityPair[] cities = listCities.toArray(new CityPair[listCities.size()]);

      float srcArrayX1[] = new float[cities.length];
      float srcArrayX2[] = new float[cities.length];
      float srcArrayY1[] = new float[cities.length];
      float srcArrayY2[] = new float[cities.length];
      float dstArray[] = new float[cities.length];
      for (int i = 0; i < cities.length; i++) {
         srcArrayX1[i] = cities[i].getX1();
         srcArrayX2[i] = cities[i].getX2();
         srcArrayY1[i] = cities[i].getY1();
         srcArrayY2[i] = cities[i].getY2();
      }

      Pointer srcX1 = Pointer.to(srcArrayX1);
      Pointer srcX2 = Pointer.to(srcArrayX2);
      Pointer srcY1 = Pointer.to(srcArrayY1);
      Pointer srcY2 = Pointer.to(srcArrayY2);
      Pointer dst = Pointer.to(dstArray);

      // The platform, device type and device number
      // that will be used
      final int platformIndex = 0;
      final long deviceType = CL_DEVICE_TYPE_GPU;
      final int deviceIndex = 0;
      long initial = System.nanoTime();
      // Enable exceptions and subsequently omit error checks in this sample
      CL.setExceptionsEnabled(true);

      // Obtain the number of platforms
      int numPlatformsArray[] = new int[1];
      clGetPlatformIDs(0, null, numPlatformsArray);
      int numPlatforms = numPlatformsArray[0];

      // Obtain a platform ID
      cl_platform_id platforms[] = new cl_platform_id[numPlatforms];
      clGetPlatformIDs(platforms.length, platforms, null);
      cl_platform_id platform = platforms[platformIndex];

      // Initialize the context properties
      cl_context_properties contextProperties = new cl_context_properties();
      contextProperties.addProperty(CL_CONTEXT_PLATFORM, platform);

      // Obtain the number of devices for the platform
      int numDevicesArray[] = new int[1];
      clGetDeviceIDs(platform, deviceType, 0, null, numDevicesArray);
      int numDevices = numDevicesArray[0];

      // Obtain a device ID 
      cl_device_id devices[] = new cl_device_id[numDevices];
      clGetDeviceIDs(platform, deviceType, numDevices, devices, null);
      cl_device_id device = devices[deviceIndex];

      // Create a context for the selected device
      cl_context context = clCreateContext(
              contextProperties, 1, new cl_device_id[]{device},
              null, null, null);

      // Create a command-queue for the selected device
      cl_queue_properties properties = new cl_queue_properties();
      cl_command_queue commandQueue = clCreateCommandQueueWithProperties(
              context, device, properties, null);

      // Allocate the memory objects for the input- and output data
      cl_mem srcMemX1 = clCreateBuffer(context,
              CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
              Sizeof.cl_float * cities.length, srcX1, null);
      cl_mem srcMemX2 = clCreateBuffer(context,
              CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
              Sizeof.cl_float * cities.length, srcX2, null);
      cl_mem srcMemY1 = clCreateBuffer(context,
              CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
              Sizeof.cl_float * cities.length, srcY1, null);
      cl_mem srcMemY2 = clCreateBuffer(context,
              CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
              Sizeof.cl_float * cities.length, srcY2, null);
      cl_mem dstMem = clCreateBuffer(context,
              CL_MEM_READ_WRITE,
              Sizeof.cl_float * cities.length, null, null);

      // Create the program from the source code
      cl_program program = clCreateProgramWithSource(context,
              1, new String[]{programSource}, null, null);

      // Build the program
      clBuildProgram(program, 0, null, null, null, null);

      // Create the kernel
      cl_kernel kernel = clCreateKernel(program, "sampleKernel", null);

      // Set the arguments for the kernel
      int a = 0;
      clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(srcMemX1));
      clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(srcMemX2));
      clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(srcMemY1));
      clSetKernelArg(kernel, 3, Sizeof.cl_mem, Pointer.to(srcMemY2));
      clSetKernelArg(kernel, 4, Sizeof.cl_mem, Pointer.to(dstMem));

      // Set the work-item dimensions
      long global_work_size[] = new long[]{cities.length};

      // Execute the kernel
      clEnqueueNDRangeKernel(commandQueue, kernel, 1, null,
              global_work_size, null, 0, null, null);

      // Read the output data
      clEnqueueReadBuffer(commandQueue, dstMem, CL_TRUE, 0,
              cities.length * Sizeof.cl_float, dst, 0, null, null);

      // Release kernel, program, and memory objects
      clReleaseMemObject(srcMemX1);
      clReleaseMemObject(srcMemX2);
      clReleaseMemObject(srcMemY1);
      clReleaseMemObject(srcMemY2);
      clReleaseMemObject(dstMem);
      clReleaseKernel(kernel);
      clReleaseProgram(program);
      clReleaseCommandQueue(commandQueue);
      clReleaseContext(context);
      long end = System.nanoTime();

      System.out.println(end - initial);
//        double distanceTravelled = 0.0;
//        for (float f : dstArray) {
//            distanceTravelled += f;
//        }
//        System.out.println("TEMPO\t" + (System.nanoTime() - a) + "\tDISTANCIA\t" + distanceTravelled);
//        this.distance = (int) distanceTravelled;
      return dstArray;
   }

   private void evolve(int g) {
      double[] selectionP = new double[N];// ѡ�����
      double sum = 0.0;
      double tmp = 0.0;

      for (int i = 0; i < N; i++) {
         double localFitness = chromosomes[i].getFitness();
         sum += localFitness;
         if (localFitness > bestFitness) {
            bestFitness = localFitness;
            bestLength = (int) (1.0 / bestFitness);
            for (int j = 0; j < cityNum; j++) {
               bestTour[j] = chromosomes[i].getTour()[j];
            }

         }
      }
      averageFitness[g] = sum / N;

//        System.out.println("The average fitness in " + g + " generation is: " + averageFitness[g]
//                + ", and the best fitness is: " + bestFitness);
      for (int i = 0; i < N; i++) {
         tmp += chromosomes[i].getFitness() / sum;
         selectionP[i] = tmp;
      }
      Random random = new Random(917693407000L);
      for (int i = 0; i < N; i = i + 2) {

         Chromosome[] children = new Chromosome[2];

         for (int j = 0; j < 2; j++) {

            int selectedCity = 0;
            for (int k = 0; k < N - 1; k++) {
               double p = random.nextDouble();
               if (p > selectionP[k] && p <= selectionP[k + 1]) {
                  selectedCity = k;
               }
               if (k == 0 && random.nextDouble() <= selectionP[k]) {
                  selectedCity = 0;
               }
            }
            try {
               children[j] = (Chromosome) chromosomes[selectedCity].clone();
            } catch (CloneNotSupportedException e) {
            }
         }
         if (random.nextDouble() < p_c_t) {
            int cutPoint1 = -1;
            int cutPoint2 = -1;
            int r1 = random.nextInt(cityNum);
            if (r1 > 0 && r1 < cityNum - 1) {
               cutPoint1 = r1;
               int r2 = random.nextInt(cityNum - r1);
               if (r2 == 0) {
                  cutPoint2 = r1 + 1;
               } else if (r2 > 0) {
                  cutPoint2 = r1 + r2;
               }

            }
            if (cutPoint1 > 0 && cutPoint2 > 0) {
               int[] tour1 = new int[cityNum];
               int[] tour2 = new int[cityNum];
               if (cutPoint2 == cityNum - 1) {
                  for (int j = 0; j < cityNum; j++) {
                     tour1[j] = children[0].getTour()[j];
                     tour2[j] = children[1].getTour()[j];
                  }
               } else {
                  for (int j = 0; j < cityNum; j++) {
                     if (j < cutPoint1) {
                        tour1[j] = children[0].getTour()[j];
                        tour2[j] = children[1].getTour()[j];
                     } else if (j >= cutPoint1 && j < cutPoint1 + cityNum - cutPoint2 - 1) {
                        tour1[j] = children[0].getTour()[j + cutPoint2 - cutPoint1 + 1];
                        tour2[j] = children[1].getTour()[j + cutPoint2 - cutPoint1 + 1];
                     } else {
                        tour1[j] = children[0].getTour()[j - cityNum + cutPoint2 + 1];
                        tour2[j] = children[1].getTour()[j - cityNum + cutPoint2 + 1];
                     }

                  }
               }

               for (int j = 0; j < cityNum; j++) {
                  if (j < cutPoint1 || j > cutPoint2) {

                     children[0].getTour()[j] = -1;
                     children[1].getTour()[j] = -1;
                  } else {
                     int tmp1 = children[0].getTour()[j];
                     children[0].getTour()[j] = children[1].getTour()[j];
                     children[1].getTour()[j] = tmp1;
                  }
               }
               if (cutPoint2 == cityNum - 1) {
                  int position = 0;
                  for (int j = 0; j < cutPoint1; j++) {
                     for (int m = position; m < cityNum; m++) {
                        boolean flag = true;
                        for (int n = 0; n < cityNum; n++) {
                           if (tour1[m] == children[0].getTour()[n]) {
                              flag = false;
                              break;
                           }
                        }
                        if (flag) {

                           children[0].getTour()[j] = tour1[m];
                           position = m + 1;
                           break;
                        }
                     }
                  }
                  position = 0;
                  for (int j = 0; j < cutPoint1; j++) {
                     for (int m = position; m < cityNum; m++) {
                        boolean flag = true;
                        for (int n = 0; n < cityNum; n++) {
                           if (tour2[m] == children[1].getTour()[n]) {
                              flag = false;
                              break;
                           }
                        }
                        if (flag) {
                           children[1].getTour()[j] = tour2[m];
                           position = m + 1;
                           break;
                        }
                     }
                  }

               } else {

                  int position = 0;
                  for (int j = cutPoint2 + 1; j < cityNum; j++) {
                     for (int m = position; m < cityNum; m++) {
                        boolean flag = true;
                        for (int n = 0; n < cityNum; n++) {
                           if (tour1[m] == children[0].getTour()[n]) {
                              flag = false;
                              break;
                           }
                        }
                        if (flag) {
                           children[0].getTour()[j] = tour1[m];
                           position = m + 1;
                           break;
                        }
                     }
                  }
                  for (int j = 0; j < cutPoint1; j++) {
                     for (int m = position; m < cityNum; m++) {
                        boolean flag = true;
                        for (int n = 0; n < cityNum; n++) {
                           if (tour1[m] == children[0].getTour()[n]) {
                              flag = false;
                              break;
                           }
                        }
                        if (flag) {
                           children[0].getTour()[j] = tour1[m];
                           position = m + 1;
                           break;
                        }
                     }
                  }

                  position = 0;
                  for (int j = cutPoint2 + 1; j < cityNum; j++) {
                     for (int m = position; m < cityNum; m++) {
                        boolean flag = true;
                        for (int n = 0; n < cityNum; n++) {
                           if (tour2[m] == children[1].getTour()[n]) {
                              flag = false;
                              break;
                           }
                        }
                        if (flag) {
                           children[1].getTour()[j] = tour2[m];
                           position = m + 1;
                           break;
                        }
                     }
                  }
                  for (int j = 0; j < cutPoint1; j++) {
                     for (int m = position; m < cityNum; m++) {
                        boolean flag = true;
                        for (int n = 0; n < cityNum; n++) {
                           if (tour2[m] == children[1].getTour()[n]) {
                              flag = false;
                              break;
                           }
                        }
                        if (flag) {
                           children[1].getTour()[j] = tour2[m];
                           position = m + 1;
                           break;
                        }
                     }
                  }
               }

            }
         }
         if (random.nextDouble() < p_m_t) {
            for (int j = 0; j < 2; j++) {
               int cutPoint1 = -1;
               int cutPoint2 = -1;
               int r1 = random.nextInt(cityNum);
               if (r1 > 0 && r1 < cityNum - 1) {
                  cutPoint1 = r1;
                  int r2 = random.nextInt(cityNum - r1);
                  if (r2 == 0) {
                     cutPoint2 = r1 + 1;
                  } else if (r2 > 0) {
                     cutPoint2 = r1 + r2;
                  }
               }
               if (cutPoint1 > 0 && cutPoint2 > 0) {
                  List<Integer> tour = new ArrayList<>();
                  if (cutPoint2 == cityNum - 1) {
                     for (int k = 0; k < cutPoint1; k++) {
                        tour.add(children[j].getTour()[k]);
                     }
                  } else {
                     for (int k = 0; k < cityNum; k++) {
                        if (k < cutPoint1 || k > cutPoint2) {
                           tour.add(children[j].getTour()[k]);
                        }
                     }
                  }
                  int position = random.nextInt(tour.size());

                  if (position == 0) {
                     for (int k = cutPoint2; k >= cutPoint1; k--) {
                        tour.add(0, children[j].getTour()[k]);
                     }
                  } else if (position == tour.size() - 1) {
                     for (int k = cutPoint1; k <= cutPoint2; k++) {
                        tour.add(children[j].getTour()[k]);
                     }
                  } else {

                     for (int k = cutPoint1; k <= cutPoint2; k++) {
                        tour.add(position, children[j].getTour()[k]);
                     }
                  }
                  for (int k = 0; k < cityNum; k++) {
                     children[j].getTour()[k] = tour.get(k);
                  }
               }

            }
         }
         nextGeneration[i] = children[0];
         nextGeneration[i + 1] = children[1];

      }

      for (int k = 0; k < N; k++) {
         try {
            chromosomes[k] = (Chromosome) nextGeneration[k].clone();
         } catch (CloneNotSupportedException e) {
         }
      }
   }

   private void printOptimal() {
      System.out.println("The best fitness is: " + bestFitness);
      System.out.println("The best tour length is: " + bestLength);
      System.out.println("The best tour is: ");

      System.out.print(bestTour[0]);
      for (int i = 1; i < cityNum; i++) {
         System.out.print("->" + bestTour[i]);
      }
      System.out.println();
   }

   private void outputResults() {
      String filename = "result.txt";
      /*
		 * File file = new File(filename); if (!file.exists()) { try {
		 * file.createNewFile(); } catch (IOException e) { // TODO Auto-generated catch
		 * block e.printStackTrace(); } }
       */
      try {
         @SuppressWarnings("resource")
         FileOutputStream outputStream = new FileOutputStream(filename);
         for (int i = 0; i < averageFitness.length; i++) {
            String line = String.valueOf(averageFitness[i]) + "\r\n";

            outputStream.write(line.getBytes());

         }

      } catch (FileNotFoundException e) {
         // TODO Auto-generated catch block
         e.printStackTrace();
      } catch (IOException e) {
         // TODO Auto-generated catch block
         e.printStackTrace();
      }

   }

   public Chromosome[] getChromosomes() {
      return chromosomes;
   }

   public void setChromosomes(Chromosome[] chromosomes) {
      this.chromosomes = chromosomes;
   }

   public int getCityNum() {
      return cityNum;
   }

   public void setCityNum(int cityNum) {
      this.cityNum = cityNum;
   }

   public double getP_c_t() {
      return p_c_t;
   }

   public void setP_c_t(double p_c_t) {
      this.p_c_t = p_c_t;
   }

   public double getP_m_t() {
      return p_m_t;
   }

   public void setP_m_t(double p_m_t) {
      this.p_m_t = p_m_t;
   }

   public int getMAX_GEN() {
      return MAX_GEN;
   }

   public void setMAX_GEN(int mAX_GEN) {
      MAX_GEN = mAX_GEN;
   }

   public int getBestLength() {
      return bestLength;
   }

   public void setBestLength(int bestLength) {
      this.bestLength = bestLength;
   }

   public int[] getBestTour() {
      return bestTour;
   }

   public void setBestTour(int[] bestTour) {
      this.bestTour = bestTour;
   }

   public double[] getAverageFitness() {
      return averageFitness;
   }

   public void setAverageFitness(double[] averageFitness) {
      this.averageFitness = averageFitness;
   }

   public int getN() {
      return N;
   }

   public void setN(int n) {
      N = n;
   }

   public int[][] getDistance() {
      return distance;
   }

   public void setDistance(int[][] distance) {
      this.distance = distance;
   }

   /**
    * @param args
    * @throws IOException
    */
   public static void main(String[] args) throws IOException {

//        GA_JOCL ga = new GA_JOCL(100, 48, 1000, 0.95, 0.5, "src/resources/att48.tsp");
//        GA_JOCL ga = new GA_JOCL(100, 575, 1000, 0.95, 0.5, "src/resources/rat575.tsp");
//        GA_JOCL ga = new GA_JOCL(100, 1379, 1000, 0.95, 0.5, "src/resources/nrw1379.tsp");
      GA_JOCL ga = new GA_JOCL(100, 14051, 1000, 0.95, 0.5, "src/resources/brd14051.tsp");
      ga.solve();
   }

}
