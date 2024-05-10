#include <iostream>
#include <cmath>
#include <random>
#include <assert.h>

float relu(float x) {return x>0?x:0;}
float dx_relu(float x){return x>0?1:0;}//x in this case is the output of relu, nasty way to do derivatives, but oh well.

float random_float()
{
    static std::default_random_engine e;
    static std::uniform_real_distribution<> dis(0, 1); // range [0, 1)
    return dis(e);
}

template< int num_classes >
struct LeNet{
    static_assert( num_classes > 0 );
    
    float l1_filters[6][5][5];
  //float l2_filters... actually just use a 2x2 max pool filter
  //float l3_filters[16][5][5]; actually use different sized filters for each. 6 3x5x5, 9 4x5x5, & 1 6x5x5
    float l3_3_filters[6][3][5][5];
    float l3_4_filters[9][4][5][5];
    float l3_6_filters[1][6][5][5];
  //floatl4_filters... actually just use a 2x2 max pool filter
    float l5_filter[400][120];
    float l6_filter[120][84];
    float l7_filter[84][num_classes];
    
    float l1_outputs[6][28][28];
    float l2_outputs[6][14][14];
    float l3_outputs[16][10][10];
    float l4_outputs[16][5][5];
    float l5_outputs[120];
    float l6_outputs[84];
    float l7_outputs[num_classes];
    
    float l1_delta[6][28][28];
    float l2_delta[6][14][14];
    float l3_delta[16][10][10];
    float l4_delta[16][5][5];
    float l5_delta[120];
    float l6_delta[84];
    float l7_delta[num_classes];
    
    float soft_maximum,soft_sum;
    
    uint32_t idx_3s[6][3] = {
                            0,1,2,
                            1,2,3,
                            2,3,4,
                            3,4,5,
                            4,5,0,
                            5,0,1,
                            };
    uint32_t idx_4s[9][4] = {
                            0,1,2,3,
                            1,2,3,4,
                            2,3,4,5,
                            3,4,5,0,
                            4,5,0,1,
                            5,0,1,2,
                            0,1,3,4,
                            1,2,4,5,
                            0,2,3,5
                            };
    uint32_t idx_6s[1][6] = {
                            0,1,2,3,4,5
                            };
    
    float l1_bias[6][28][28];
    float l3_bias[16][10][10];
    float l5_bias[120];
    float l6_bias[84];
    
    int forward_pass(float input[32][32]){
        
        /// Layer ONE Forward Propagation
        /// 2D Convolution With 6 Filters
        for( int i = 0; i < 6; i++ ){
            for( int j = 0; j < 28; j++ ){
                for( int k = 0; k < 28; k++ ){
                    float pixel = 0;
                    for( int x = 0; x < 5; x++ ){
                        for( int y = 0; y < 5; y++ ){
                            pixel += input[j+x][k+y] * l1_filters[i][x][y];
                        }
                    }
                    l1_outputs[i][j][k] = relu( pixel+l1_bias[i][j][k] );
                }
            }
        }
        
        /// Layer TWO Forward Propagation
        /// This is Max Pooling on 6 2D outputs
        for( int i = 0; i < 6; i++ ){
            for( int j = 0; j < 14; j++ ){
                for( int k = 0; k < 14; k++ ){
                    float max_1 = std::max(l1_outputs[i][j*2  ][k*2],l1_outputs[i][j*2  ][k*2+1]);
                    float max_2 = std::max(l1_outputs[i][j*2+1][k*2],l1_outputs[i][j*2+1][k*2+1]);
                    l2_outputs[i][j][k] = std::max( max_1, max_2 );
                }
            }
        }
        
        
        /// Layer THREE Forward Propagation
        /// This is 3D Convolution with 16 Filters
        
        /// 6 Filters have 3 layers
        
        for( int i = 0; i < 6; i++ ){
            for( int j = 0; j < 10; j++ ){
                for( int k = 0; k < 10; k++ ){
                    float pixel = 0;
                    for( int l = 0; l < 3; l++ ){
                        for( int x = 0; x < 5; x++ ){
                            for( int y = 0; y < 5; y++ ){
                                pixel += l2_outputs[ idx_3s[i][l] ][j+x][k+y]*l3_3_filters[i][l][x][y];
                            }
                        }
                    }
                    l3_outputs[i][j][k] = relu( pixel+l3_bias[i][j][k] );
                }
            }
        }
        
        /// 9 Filters have 4 layers
        
        for( int i = 0; i < 9; i++ ){
            for( int j = 0; j < 10; j++ ){
                for( int k = 0; k < 10; k++ ){
                    float pixel = 0;
                    for( int l = 0; l < 4; l++ ){
                        for( int x = 0; x < 5; x++ ){
                            for( int y = 0; y < 5; y++ ){
                                pixel += l2_outputs[ idx_4s[i][l] ][j+x][k+y]*l3_4_filters[i][l][x][y];//basic convolution.
                            }
                        }
                    }
                    l3_outputs[i+6][j][k] = relu( pixel+l3_bias[i+6][j][k] );
                }
            }
        }
        
        /// 1 Filter has 6 Layers
        
        for( int i = 0; i < 1; i++ ){
            for( int j = 0; j < 10; j++ ){
                for( int k = 0; k < 10; k++ ){
                    float pixel = 0;
                    for( int l = 0; l < 6; l++ ){
                        for( int x = 0; x < 5; x++ ){
                            for( int y = 0; y < 5; y++ ){
                                pixel += l2_outputs[ idx_6s[i][l] ][j+x][k+y]*l3_6_filters[i][l][x][y];//basic convolution.
                            }
                        }
                    }
                    l3_outputs[i+15][j][k] = relu( pixel+l3_bias[i+15][j][k] );
                }
            }
        }
        
        /// Layer FOUR Max Pooling
        
        for( int i = 0; i < 16; i++ ){
            for( int j = 0; j < 5; j++ ){
                for( int k = 0; k < 5; k++ ){
                    float max_1 = std::max(l3_outputs[i][j*2  ][k*2],l3_outputs[i][j*2  ][k*2+1]);
                    float max_2 = std::max(l3_outputs[i][j*2+1][k*2],l3_outputs[i][j*2+1][k*2+1]);
                    l4_outputs[i][j][k] = std::max( max_1, max_2 );
                }
            }
        }
        
        /// Layer FIVE MLP
        
        for( int i = 0; i < 120; i++ ){
            float pixel = 0;
            for( int j = 0; j < 400; j++ ){
                pixel += ((float*)l4_outputs)[j]*l5_filter[j][i];
            }
            l5_outputs[i] = relu( pixel+l5_bias[i] );
        }
        
        /// LAYER SIX MLP
        for( int i = 0; i < 84; i++ ){
            float pixel = 0;
            for( int j = 0; j < 120; j++ ){
                pixel += l5_outputs[j]*l6_filter[j][i];
            }
            l6_outputs[i] = relu( pixel+l6_bias[i] );
        }
        
        /// Layer SEVEN Forward Propagation
        /// SOFTMAX!!!
        
        for( int i = 0; i < num_classes; i++ ){
            float pixel = 0;
            for( int j = 0; j < 84; j++ ){
                pixel += l6_outputs[j]*l7_filter[j][i];
            }
            l7_outputs[i] = relu( pixel );
        }
        
        int prediction = -1;
        
        soft_maximum = l7_outputs[0];
        for( int i = 0; i < num_classes; i++ )
            soft_maximum = std::max( soft_maximum, l7_outputs[i] );
            
        soft_sum = 0;
        for( int i = 0; i < num_classes; i++ ){
            if( l7_outputs[i] == soft_maximum ) prediction = i;
            l7_outputs[i] = std::exp(l7_outputs[i]-soft_maximum);
            soft_sum += l7_outputs[i];
        }
        
        for( int i = 0; i < num_classes; i++ )
            l7_outputs[i] /= soft_sum;
        
        assert( prediction >= 0 );
        return prediction;
    }
    
    void backward_pass(float correct_output[num_classes], float speed,float input[32][32]){
        
        /// Layer SEVEN Backward Propagation
        
        for( int i = 0; i < num_classes; i++ ){
            l7_delta[i] = (correct_output[i]-l7_outputs[i]);
            
            assert( std::isfinite( l7_delta[i] ) );
        }
        
        for( int i = 0; i < 84; i++ ){
                float sum = 0;
            for( int j = 0; j < num_classes; j++ ){
                sum -= l7_delta[j]*l7_filter[i][j];
                l7_filter[i][j] += speed*l7_delta[j]*l6_outputs[i];
            }
            l6_delta[i] = sum*dx_relu(l6_outputs[i]);
        }
        
        /// Layer SIX Backwards Propagation
        
        for( int i = 0; i < 120; i++ ){
                float sum = 0;
            for( int j = 0; j < 84; j++ ){
                sum += l6_delta[j]*l6_filter[i][j];
                l6_filter[i][j] -= speed*l6_delta[j]*l5_outputs[i];
            }
            l6_bias[i] -= speed*l6_delta[i]*l5_outputs[i];
            l5_delta[i] = sum*dx_relu(l5_outputs[i]);
        }
        
        /// Layer FIVE Backwards Propagation
        
        for( int i = 0; i < 400; i++ ){
                float sum = 0;
            for( int j = 0; j < 120; j++ ){
                sum += l5_delta[j]*l5_filter[i][j];
                l5_filter[i][j] -= speed*l5_delta[j]*((float*)l4_outputs)[i];
            }
            l5_bias[i] -= speed*l5_delta[i]*((float*)l4_outputs)[i];
            ( (float*) l4_delta)[i] = sum;
        }
        
        /// Layer FOUR Backwards Propagation
        /// Since layer two is just a max pool, we just need to set the delta appropriately for the previous layer.
        /// This is done by counting the maximums, and dividing the delta evenly between them.
        
        for( int i = 0; i < 16; i++ ){
            for( int j = 0; j < 5; j++ ){
                for( int k = 0; k < 5; k++ ){
                    int max_count = 0;
                    
                    if( l3_outputs[i][2*j  ][2*k  ] == l4_outputs[i][j][k] )
                        max_count++;
                    if( l3_outputs[i][2*j  ][2*k+1] == l4_outputs[i][j][k] )
                        max_count++;
                    if( l3_outputs[i][2*j+1][2*k  ] == l4_outputs[i][j][k] )
                        max_count++;
                    if( l3_outputs[i][2*j+1][2*k+1] == l4_outputs[i][j][k] )
                        max_count++;
                    assert( max_count > 0);
                    
                    if( l3_outputs[i][2*j  ][2*k] == l4_outputs[i][j][k] )
                        l3_delta[i][2*j  ][2*k  ] = l4_delta[i][j][k]/max_count;
                        
                    if( l3_outputs[i][2*j  ][2*k+1] == l4_outputs[i][j][k] )
                        l3_delta[i][2*j  ][2*k+1] = l4_delta[i][j][k]/max_count;
                        
                    if( l3_outputs[i][2*j+1][2*k  ] == l4_outputs[i][j][k] )
                        l3_delta[i][2*j+1][2*k  ] = l4_delta[i][j][k]/max_count;
                        
                    if( l3_outputs[i][2*j+1][2*k+1] == l4_outputs[i][j][k] )
                        l3_delta[i][2*j+1][2*k+1] = l4_delta[i][j][k]/max_count;
                    
                    
                }
            }
        }
        
        /// Layer THREE Backwards Propagation
        /// 3D convolution with 3 different filter sizes and 16 filters
        
        for( int i = 0; i < 6*14*14; i++ )
            ((float*)l2_delta)[i] = 0;// erase previous delta so we can update it in steps
        
        
        for( int i = 0; i < 9; i++ ){
            for( int j = 0; j < 10; j++ ){
                for( int k = 0; k < 10; k++ ){
                    float delta_sum = 0;
                    for( int l = 0; l < 3; l++ ){
                        for( int x = 0; x < 5; x++ ){
                            for( int y = 0; y < 5; y++ ){
                                delta_sum += l3_3_filters[i][l][x][y]*l3_delta[i][j][k];
                                l3_3_filters[i][l][x][y] -= speed*l3_delta[i][j][k]*l2_outputs[idx_3s[i][l]][j+x][k+y];
                            }
                        }
                        l2_delta[   idx_3s[i][l]   ][j][k] += delta_sum*dx_relu( l3_outputs[i][j][k] );
                    }
                    l3_bias[i][j][k] -= speed*l3_delta[i][j][k];
                }
            }
        }
        
        for( int i = 0; i < 9; i++ ){
            for( int j = 0; j < 10; j++ ){
                for( int k = 0; k < 10; k++ ){
                    float delta_sum = 0;
                    for( int l = 0; l < 4; l++ ){
                        for( int x = 0; x < 5; x++ ){
                            for( int y = 0; y < 5; y++ ){
                                delta_sum += l3_4_filters[i][l][x][y]*l3_delta[i+6][j][k];
                                l3_4_filters[i][l][x][y] -= speed*l3_delta[i+6][j][k]*l2_outputs[idx_4s[i][l]][j+x][k+y];
                            }
                        }
                        l2_delta[idx_4s[i][l]][j][k] += delta_sum*dx_relu( l3_outputs[i][j][k] );
                    }
                    l3_bias[i+6][j][k] -= speed*l3_delta[i+6][j][k];
                }
            }
        }
        
        for( int i = 0; i < 1; i++ ){
            for( int j = 0; j < 10; j++ ){
                for( int k = 0; k < 10; k++ ){
                    float delta_sum = 0;
                    for( int l = 0; l < 6; l++ ){
                        for( int x = 0; x < 5; x++ ){
                            for( int y = 0; y < 5; y++ ){
                                delta_sum += l3_4_filters[i][l][x][y]*l3_delta[i+15][j][k];
                                l3_4_filters[i][l][x][y] -= speed*l3_delta[i+15][j][k]*l2_outputs[idx_6s[i][l]][j+x][k+y];
                            }
                        }
                        l2_delta[idx_6s[i][l]][j][k] += delta_sum*dx_relu( l3_outputs[i][j][k] );
                    }
                    l3_bias[i+15][j][k] -= speed*l3_delta[i+15][j][k];
                }
            }
        }
        
        /// Layer TWO Backwards Propagation
        /// Since layer two is just a max pool, we just need to set the delta appropriately for the previous layer.
        /// This is done by counting the maximums, and dividing the delta evenly between them.
        for( int i = 0; i < 6; i++ ){
            for( int j = 0; j < 14; j++ ){
                for( int k = 0; k < 14; k++ ){
                    int max_count = 0;
                    
                    if( l1_outputs[i][2*j  ][2*k] == l2_outputs[i][j][k] )
                        max_count++;
                    if( l1_outputs[i][2*j  ][2*k+1] == l2_outputs[i][j][k] )
                        max_count++;
                    if( l1_outputs[i][2*j+1][2*k  ] == l2_outputs[i][j][k] )
                        max_count++;
                    if( l1_outputs[i][2*j+1][2*k+1] == l2_outputs[i][j][k] )
                        max_count++;
                    assert( max_count > 0);
                    
                    if( l1_outputs[i][2*j  ][2*k] == l2_outputs[i][j][k] )
                        l1_delta[i][2*j  ][2*k  ] = l2_delta[i][j][k]/max_count;
                        
                    if( l1_outputs[i][2*j  ][2*k+1] == l2_outputs[i][j][k] )
                        l1_delta[i][2*j  ][2*k+1] = l2_delta[i][j][k]/max_count;
                        
                    if( l1_outputs[i][2*j+1][2*k  ] == l2_outputs[i][j][k] )
                        l1_delta[i][2*j+1][2*k  ] = l2_delta[i][j][k]/max_count;
                        
                    if( l1_outputs[i][2*j+1][2*k+1] == l2_outputs[i][j][k] )
                        l1_delta[i][2*j+1][2*k+1] = l2_delta[i][j][k]/max_count;
                    
                    
                }
            }
        }
        
        /// Layer ONE Backwards Propagation
        /// Essentially convolution in reverse, sorta, with derivatives.
        for( int i = 0; i < 6; i++ ){
            for( int j = 0; j < 28; j++ ){
                for( int k = 0; k < 28; k++ ){
                    
                    for( int x = 0; x < 5; x++ ){
                        for( int y = 0; y < 5; y++ ){
                            l1_filters[i][x][y] -= speed*l1_delta[i][j][k]*input[j+x][k+y];
                        }
                    }
                    
                    l1_bias[i][j][k] -= speed*l1_delta[i][j][k];
                    
                }
            }
        }
    }
    
    void fourier_pass(){
        
    }
    
    LeNet(){
    
    for( int i = 0; i < 6*5*5; i++ )
        ( (float*) l1_filters )[i] = random_float();
    for( int i = 0; i < 6*3*5*5; i++ )
        ( (float*) l3_3_filters )[i] = random_float();
    for( int i = 0; i < 9*4*5*5; i++ )
        ( (float*) l3_4_filters )[i] = random_float();
    for( int i = 0; i < 1*6*5*5; i++ )
        ( (float*) l3_6_filters )[i] = random_float();
    for( int i = 0; i < 400*120; i++ )
        ( (float*) l5_filter )[i] = random_float();
    for( int i = 0; i < 120*84; i++ )
        ( (float*) l6_filter )[i] = random_float();
    for( int i = 0; i < 84*num_classes; i++ )
        ( (float*) l7_filter )[i] = random_float();
    
    
    for( int i = 0; i < 6*28*28; i++ )
        ( (float*) l1_bias )[i] = random_float();
    for( int i = 0; i < 16*10*10; i++ )
        ( (float*) l3_bias )[i] = random_float();
    for( int i = 0; i < 120; i++ )
        ( (float*) l5_bias )[i] = random_float();
    for( int i = 0; i < 84; i++ )
        ( (float*) l6_bias )[i] = random_float();
    }
    
};




int main(int argc, char** argv){
    
    float input[32][32];
    for( int i = 0; i < 32; i++ ){
        for( int j = 0; j < 32; j++ ){
            input[i][j] = random_float();
        }
    }
    
    LeNet<3> model1;
    
    float correct_output[3] = {0,0,0};
    
    for( int i = 0; i < 5; i++ ){
        model1.forward_pass(input);
        model1.backward_pass(correct_output,.2,input);
    }
    return model1.forward_pass(input);
}
