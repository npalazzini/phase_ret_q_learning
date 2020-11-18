import phase_ret_algs as alg
import numpy as np

np.random.seed(1997)

alg.init(8)

def get_dim(filename):

    f = open(filename, 'r')
    fl = f.readline()
    sl = f.readline()
    w1,w2=sl.split()
    x_dim=int(w1)
    y_dim=int(w2)

    f.close()

    return x_dim, y_dim

def print_modulus_raw(comp_data, filename, bits):

    data=np.absolute(comp_data)
    ngrey = np.power(2,bits)-1
    x_dim, y_dim=data.shape
    head="P2\n"+str(x_dim)+" "+str(y_dim)+"\n"+str(ngrey)
    max=np.amax(data)
    if max==0:
        max=1

    data=np.absolute(1.*ngrey*data/max).astype(int)

    np.savetxt(filename, data, header=head, comments='', fmt='%i')


class environment:
    def __init__(self, HIO_ITERATIONS, ER_ITERATIONS, SQUARE_ROOT, MASK, POISSON, photons, IMPOSE_REALITY, HIO_BETA, R_COEFF, num_actions, error_target):
        self.HIO_ITERATIONS=HIO_ITERATIONS
        self.ER_ITERATIONS=ER_ITERATIONS
        self.IMPOSE_REALITY=IMPOSE_REALITY
        self.HIO_BETA=HIO_BETA
        self.num_actions=num_actions #can be useful in action design
        self.error_target=error_target

        print("Reading data...")

        x_dim,y_dim=get_dim("INPUT/intensities.pgm")

        self.intensities=np.loadtxt("INPUT/intensities.pgm", skiprows=3)
        self.support=np.loadtxt("INPUT/support.pgm", skiprows=3).astype("int32")
        self.input_data=np.loadtxt("INPUT/density.pgm", skiprows=3)

        self.intensities=self.intensities.reshape((x_dim,y_dim))
        self.support=self.support.reshape((x_dim,y_dim))
        self.input_data=self.input_data.reshape((x_dim,y_dim))

        if MASK:
            self.mask=np.loadtxt("INPUT/mask.pgm", skiprows=3)
            self.mask=self.mask.reshape((x_dim,y_dim))

        print("Dimensions:", x_dim, "x", y_dim)

        sigma=self.support.size/np.count_nonzero(self.support)

        print("Over-sampling ratio:",sigma)

        print("Setting up the pattern ...")

        if POISSON:
            max_val=np.max(self.intensities)
            self.intensities=self.intensities/max_val*photons
            for i in range(self.intensities.shape[0]):
                for j in range(self.intensities.shape[1]):
                    self.intensities[i][j]=np.random.poisson(self.intensities[i][j])

        print_modulus_raw(np.log(1+self.intensities),"OUTPUT/start_intensities.pgm", 8)

        if MASK:
            for i in range(self.mask.shape[0]):
                for j in range(self.mask.shape[1]):
                    if self.mask[i][j]>0:
                        self.intensities[i][j]=-1

        if SQUARE_ROOT:
            for i in range(self.intensities.shape[0]):
                for j in range(self.intensities.shape[1]):
                    if self.intensities[i][j]>=0:
                         self.intensities[i][j]=np.sqrt(self.intensities[i][j])

        self.intensities=np.fft.fftshift(self.intensities)

        print("Setting up the support ...")

        for i in range(self.support.shape[0]):
            for j in range(self.support.shape[1]):
                if self.support[i][j]>0:
                    self.support[i][j]=1

        self.start_support=self.support

        data=self.input_data
        data=np.fft.fft2(data)
        temp_phase = np.angle(data) + R_COEFF*(np.random.rand(*data.shape)*2*np.pi-np.pi)
        data=self.intensities*np.exp(1j*temp_phase)
        self.data_start=np.fft.ifft2(data)


    def _get_error(self):
        temp_data=self.data
        temp_data=alg.HIO(self.intensities, self.start_support, temp_data, 20 , self.HIO_BETA, self.IMPOSE_REALITY)
        temp_data=alg.ER(self.intensities, self.start_support, temp_data, 10 , self.IMPOSE_REALITY)
        error=alg.get_error(temp_data, self.start_support, self.intensities)

        return error

    def _act(self, action):

        #action definition

        #if action==0:
            #do nothing (.,.)
        if action==1:
            self.tau+=0.01 #(.,+)
        if action==2:
            self.tau-=0.01 #(.,-)
        if action==3:
            self.sigma+=0.1 #(+,.)
        if action==4:
            self.sigma+=0.1 #(+,+)
            self.tau+=0.01
        if action==5:
            self.sigma+=0.1 #(+,-)
            self.tau-=0.01
        if action==6:
            self.sigma-=0.1 #(-,.)
        if action==7:
            self.sigma-=0.1 #(-,+)
            self.tau+=0.01
        if action==8:
            self.sigma-=0.1 #(-,-)
            self.tau-=0.01

        self.sigma=np.clip(self.sigma, 1.5, 3)
        self.tau=np.clip(self.tau, 0.05, 0.20)

    def step(self, action):

        self._act(action)

        self.support=alg.ShrinkWrap(self.data, self.start_support, self.sigma, self.tau)

        self.data=alg.HIO(self.intensities, self.support, self.data, self.HIO_ITERATIONS , self.HIO_BETA, self.IMPOSE_REALITY)
        self.data=alg.ER(self.intensities, self.support, self.data, self.ER_ITERATIONS, self.IMPOSE_REALITY)

        error=self._get_error()

        done=False

        if error<self.error_target:
            reward=1 #reward definition
            self.minimum=error
            done=True
        elif error<self.minimum:
            reward=self.minimum-error #reward definition
            self.minimum=error
        else:
            reward=0 #reward definition

        state_next=np.absolute(self.data)[156:356, 156:356, np.newaxis]

        return state_next, reward, done, error


    def reset(self):
        self.data=self.data_start
        self.minimum=self._get_error()
        self.support=self.start_support

        self.sigma=2.5
        self.tau=0.1

        state=np.absolute(self.data)[156:356, 156:356, np.newaxis]

        return state
