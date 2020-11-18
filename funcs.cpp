#include "funcs.hpp"

void normalize(fftw_complex *vett, double mod, int npix){

    #pragma omp parallel for
    for(int i=0; i<npix; i++){
        vett[i][0]=vett[i][0]*1./mod;
        vett[i][1]=vett[i][1]*1./mod;
    }
};

void shift(fftw_complex* data, int x_trasl, int y_trasl, int x_dim, int y_dim){
    if(x_trasl==-1)
        x_trasl=x_dim/2;

    if(y_trasl==-1)
        y_trasl=y_dim/2;

    fftw_complex* temp= (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*x_dim*y_dim);

    for(int i=0; i<x_dim; i++){
        int ii = (i + x_trasl) % x_dim;
        for(int j=0; j<y_dim; j++){
            int jj = (j + y_trasl) % y_dim;
            temp[ii * y_dim + jj][0] = data[i * y_dim + j][0];
            temp[ii * y_dim + jj][1] = data[i * y_dim + j][1];
        }
    }

    for(int i=0; i<x_dim*y_dim; i++){
        data[i][0]=temp[i][0];
        data[i][1]=temp[i][1];
    }

    fftw_free(temp);
}

void sub_intensities(fftw_complex* data, py::array_t<double, py::array::c_style> intensities){
    py::buffer_info int_buf = intensities.request();
    double *int_ptr = (double *) int_buf.ptr;

    #pragma omp parallel for
    for(int i=0; i<int_buf.size; i++){
        if(int_ptr[i]>-1){  //se il pixel del pattern e' noto
            double m=(data[i][1]*data[i][1]+data[i][0]*data[i][0]);
            if(m>0){
                m=sqrt(m);
                data[i][0] = int_ptr[i]*data[i][0]/(m);        // write on the input the experimental module times the real ...
                data[i][1] = int_ptr[i]*data[i][1]/(m);        // and the imaginary part of exp(i x phase)
            }
            else{
                double phase;
                if(data[i][1]<0) phase=atan2(data[i][1],data[i][0]) + 2.*M_PI;
                else phase=atan2(data[i][1],data[i][0]);
                data[i][0]=int_ptr[i]*cos(phase);
                data[i][1]=int_ptr[i]*sin(phase);
            }
        }
    }
};

void apply_support_er(fftw_complex *r_space, py::array_t<int, py::array::c_style> support, bool impose_reality){
    py::buffer_info supp_buf = support.request();
    int *supp_ptr = (int *) supp_buf.ptr;

    #pragma omp parallel for
    for(int i=0; i<supp_buf.size; i++){
        r_space[i][0]=r_space[i][0]*supp_ptr[i];
        if(impose_reality)
            r_space[i][1]=0;
        else
            r_space[i][1]=r_space[i][1]*supp_ptr[i];
    }
};

void apply_support_hio(fftw_complex *r_space, py::array_t<int, py::array::c_style> support, fftw_complex *buffer_r_space, double beta, bool impose_reality){
    py::buffer_info supp_buf = support.request();
    int *supp_ptr = (int *) supp_buf.ptr;

    #pragma omp parallel for
    for(int i=0; i<supp_buf.size; i++){
        r_space[i][0]=(buffer_r_space[i][0]-beta*r_space[i][0])*(1.-supp_ptr[i]) + r_space[i][0]*supp_ptr[i];
        if(impose_reality)
            r_space[i][1]=0;
        else
            r_space[i][1]=(buffer_r_space[i][1]-beta*r_space[i][1])*(1.-supp_ptr[i]) + r_space[i][1]*supp_ptr[i];

        buffer_r_space[i][0]=r_space[i][0];
        buffer_r_space[i][1]=r_space[i][1];
    }
};

py::array_t<std::complex<double>, py::array::c_style> ER(py::array_t<double, py::array::c_style> intensities, py::array_t<int, py::array::c_style> support, py::array_t<std::complex<double>, py::array::c_style> r_space, int n_iterations, bool impose_reality){
    py::buffer_info data_buf = r_space.request();
    std::complex<double> *data_ptr = (std::complex<double> *) data_buf.ptr;

    int x_dim=data_buf.shape[0];
    int y_dim=data_buf.shape[1];

    int npix=x_dim*y_dim;

    fftw_complex* data=(fftw_complex*) fftw_malloc(sizeof(fftw_complex)*npix);

    memcpy(data, data_ptr, npix*sizeof(std::complex<double>));

    fftw_plan p2k = fftw_plan_dft_2d(x_dim, y_dim, data, data, FFTW_BACKWARD, FFTW_ESTIMATE); //direttive per andare dal diretto al reciproco
    fftw_plan p2r = fftw_plan_dft_2d(x_dim, y_dim, data, data, FFTW_FORWARD, FFTW_ESTIMATE);

    for(int i_iteration=0; i_iteration<n_iterations; i_iteration++){

        fftw_execute(p2k);                         // go in the reciprocal space
        sub_intensities(data, intensities);        // substitute experimental intensities to HIO-computed ones
        fftw_execute(p2r);                         // go back in real space
        normalize(data, npix, npix);               // normalize the obtained density (to check)
                                                   // outside HIO this is assumed, but we have to force it inside
        apply_support_er(data, support, impose_reality);           // see directly the comment in the function

    }

    auto output = py::array_t<std::complex<double>, py::array::c_style>(npix);
    py::buffer_info out_buf = output.request();
    std::complex<double> *out_ptr = (std::complex<double> *) out_buf.ptr;

    memcpy(out_ptr, data, npix*sizeof(std::complex<double>));

    fftw_destroy_plan(p2k);
    fftw_destroy_plan(p2r);
    fftw_free(data);

    output.resize({x_dim,y_dim});

    return output;
};

py::array_t<std::complex<double>, py::array::c_style> HIO(py::array_t<double, py::array::c_style> intensities, py::array_t<int, py::array::c_style> support, py::array_t<std::complex<double>, py::array::c_style> r_space, int n_iterations, double beta, bool impose_reality){
    py::buffer_info data_buf = r_space.request();
    std::complex<double> *data_ptr = (std::complex<double> *) data_buf.ptr;

    int x_dim=data_buf.shape[0];
    int y_dim=data_buf.shape[1];

    int npix=x_dim*y_dim;

    fftw_complex* data=(fftw_complex*) fftw_malloc(sizeof(fftw_complex)*npix);
    fftw_complex* buffer_r_space = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*npix);

    memcpy(data, data_ptr, npix*sizeof(std::complex<double>));
    memcpy(buffer_r_space, data_ptr, npix*sizeof(std::complex<double>));

    fftw_plan p2k = fftw_plan_dft_2d(x_dim, y_dim, data, data, FFTW_BACKWARD, FFTW_ESTIMATE); //direttive per andare dal diretto al reciproco
    fftw_plan p2r = fftw_plan_dft_2d(x_dim, y_dim, data, data, FFTW_FORWARD, FFTW_ESTIMATE);

    for(int i_iteration=0; i_iteration<n_iterations; i_iteration++){ // TO CHECK: check the possibility to use a FFT which works only with real input

        fftw_execute(p2k);                         // go in the reciprocal space
        sub_intensities(data, intensities);        // substitute experimental intensities to HIO-computed ones
        fftw_execute(p2r);                         // go back in real space
        normalize(data, npix, npix);               // normalize the obtained density (to check)

        apply_support_hio(data, support, buffer_r_space, beta, impose_reality);  // see directly the comment in the function
    }

    auto output = py::array_t<std::complex<double>, py::array::c_style>(npix);
    py::buffer_info out_buf = output.request();
    std::complex<double> *out_ptr = (std::complex<double> *) out_buf.ptr;

    memcpy(out_ptr, data, npix*sizeof(std::complex<double>));

    fftw_free(buffer_r_space);
    fftw_destroy_plan(p2k);
    fftw_destroy_plan(p2r);
    fftw_free(data);

    output.resize({x_dim,y_dim});

    return output;
};

py::array_t<int, py::array::c_style> ShrinkWrap(py::array_t<std::complex<double>, py::array::c_style> r_space, py::array_t<int, py::array::c_style> original_support, double sigma, double tau){

    py::buffer_info data_buf = r_space.request();
    std::complex<double> *data_ptr = (std::complex<double> *) data_buf.ptr;

    py::buffer_info orig_buf = original_support.request();
    int *original_ptr = (int *) orig_buf.ptr;

    int x_dim=data_buf.shape[0];
    int y_dim=data_buf.shape[1];

    int npix=x_dim*y_dim;

    fftw_complex* data=(fftw_complex*) fftw_malloc(sizeof(fftw_complex)*npix);

    memcpy(data, data_ptr, npix*sizeof(std::complex<double>));

    //module
    for(int i=0; i<npix; i++){
        data[i][0]=sqrt(data[i][0]*data[i][0]+ data[i][1]*data[i][1]);
        data[i][1]=0;
    }

    //prepare gaussian filter
    fftw_complex* gaussian=(fftw_complex*) fftw_malloc(sizeof(fftw_complex)*npix);

    double sum=0;

    for(int i=0; i<npix; i++){
        double x=i%x_dim;
        double y=(int)i/x_dim;
        gaussian[i][0]=1./(sigma*sigma*2.*M_PI)*exp(-((x-x_dim/2.)*(x-x_dim/2.)+ (y-y_dim/2.)*(y-y_dim/2.))/(2.*sigma*sigma));
        sum+=gaussian[i][0];
        gaussian[i][1]=0;
    }

    for(int i=0; i<npix; i++)
        gaussian[i][0]=gaussian[i][0]/sum;

    //prepare FT plans
    fftw_plan data_p2k = fftw_plan_dft_2d(x_dim, y_dim, data, data, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_plan data_p2r = fftw_plan_dft_2d(x_dim, y_dim, data, data, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_plan gauss_p2k = fftw_plan_dft_2d(x_dim, y_dim, gaussian, gaussian, FFTW_BACKWARD, FFTW_ESTIMATE);

    //FT
    fftw_execute(data_p2k);
    fftw_execute(gauss_p2k);

    //complex multiplication
    for(int i=0; i<npix; i++){
        double re=data[i][0]*gaussian[i][0]-data[i][1]*gaussian[i][1];
        double im=data[i][0]*gaussian[i][1]+data[i][1]*gaussian[i][0];
        data[i][0]=re;
        data[i][1]=im;
    }

    //IFT
    fftw_execute(data_p2r);

    //shift
    shift(data, -1, -1, x_dim, y_dim);

    //find max
    double max=0;
    for(int i=0; i<npix; i++){
        double val=sqrt(data[i][0]*data[i][0] + data[i][1]*data[i][1]);
        if(val>max)
            max=val;
    }

    auto output = py::array_t<int>(npix);
    py::buffer_info out_buf = output.request();
    int *out_ptr = (int *) out_buf.ptr;

    //assign output
    for(int i=0; i<npix; i++){
        out_ptr[i]=0;
        double val=sqrt(data[i][0]*data[i][0]+ data[i][1]*data[i][1]);
        if(val>tau*max)
            out_ptr[i]=1*original_ptr[i];
    }

    fftw_destroy_plan(gauss_p2k);
    fftw_destroy_plan(data_p2k);
    fftw_destroy_plan(data_p2r);
    fftw_free(data);
    fftw_free(gaussian);

    output.resize({x_dim,y_dim});

    return output;
};

double get_error(py::array_t<std::complex<double>, py::array::c_style> data, py::array_t<int, py::array::c_style> support, py::array_t<double, py::array::c_style> intensities){
    py::buffer_info buf_data = data.request();
    py::buffer_info buf_supp = support.request();
    py::buffer_info buf_int = intensities.request();
    std::complex<double> *ptr_data = (std::complex<double> *) buf_data.ptr;
    int *ptr_supp = (int *) buf_supp.ptr;
    double *ptr_int = (double *) buf_int.ptr;

    int x_dim=buf_data.shape[0];
    int y_dim=buf_data.shape[1];

    int npix = x_dim*y_dim;

    fftw_complex* local_data=(fftw_complex*) fftw_malloc(sizeof(fftw_complex)*npix);

    //memcpy
    for(int i=0; i<npix; i++){
        local_data[i][0]=ptr_supp[i]*std::real(ptr_data[i]);    // put densities in the real part
        local_data[i][1]=ptr_supp[i]*std::imag(ptr_data[i]);    // put zero in the imaginary part
    }

    fftw_plan p= fftw_plan_dft_2d(x_dim, y_dim, local_data, local_data, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(p);

    double sum=0.;
    double tot=0.;
    for(int i=0; i<npix; i++){
        if(ptr_int[i]>=0){  // no measure is available where artificially |Exp| has been set < 0
            sum += ptr_int[i];
            tot += (ptr_int[i]-sqrt(local_data[i][0]*local_data[i][0]+ local_data[i][1]*local_data[i][1]))*
                   (ptr_int[i]-sqrt(local_data[i][0]*local_data[i][0]+ local_data[i][1]*local_data[i][1]));    // tot = SUM_j | sqrt(Exp) - |FFT(real_random)| |
        }
    }

    double error=sqrt(tot)/sum;     // l'errore reale e' il rapporto tra la densita' fuori dal supporto e quella totale

    fftw_free(local_data);
    fftw_destroy_plan(p);

    return error;
};
