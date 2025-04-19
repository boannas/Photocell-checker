#include <TimerOne.h>
#include <Math.h>

#define PHOTOTRANS_PIN A0
#define TIMEWIN_SIZE 500

// FIFO BUFFER CLASS =======================================================
template <typename T>
class FIFObuf{
  private:
  // variables 
    T* _head;
    T* _tail;
    T* _buffer;
    int _numel;
    size_t _buffersize;
  
  public:
    FIFObuf(size_t buffersize)
    {
      _buffersize = buffersize;
      _buffer = new T[buffersize];
      _head = _buffer;
      _tail = _buffer;
      _numel = 0;
    }

    ~FIFObuf()
    {
      delete[] _buffer;
      _head = _buffer;
      _tail = _buffer;
      _numel = 0;
    }

    // methods
    bool push(T value)
    {
      bool ans = true;

      if (_numel != _buffersize)
      {
        *_tail = value;
        _numel++;

        if (_tail == &_buffer[_buffersize-1])
        {
          _tail = _buffer;
        }
        else
        {
          _tail++;
        }
      }
      else
      {
        ans = false;
      }

      return ans;
    }

    T pop()
    {
      T ans;

      if (_numel != 0)
      {
        ans = *_head;
        _numel--;

        if (_head == _buffer)
        {
          _head = &_buffer[_buffersize-1];
        }
        else
        {
          _head--;
        }

        return ans;
      }
      else
      {
        return T();
      }
    }

    T at(unsigned int index)
    {
      return _buffer[index];
    }

    size_t size()
    {
        return _numel;
    }

    void print()
    {
      Serial.println("--------------");
      for (int i = 0; i<_buffersize; i++)
      {
        Serial.println(_buffer[i]);
      }
      Serial.println("--------------");
    }

    bool isfull()
    {
      if (_numel == _buffersize)
      {
        return true;
      }
      else
      {
        return false;
      }
    }
};

// GLOBAL INITIALIZATION ===================================================
// Timer Interrupt variables ----------
int read_freq = 500; 
int output_freq = 1;
int output_count = 0;
int output_timestamp = read_freq/output_freq;

// Phototransistor variables ----------
// Calibration
int calibrate[50];
int calibrate_size = sizeof(calibrate) / sizeof(calibrate[0]);
int calibrate_step = 0;
bool calibrate_flag = true;
float threshold = 0;
// Measurement
float last_val = 0;
float current_val;
int npeak = 0;
FIFObuf<int> timewindow(TIMEWIN_SIZE);

// MAIN ====================================================================
void setup() {
  // Serial initialization for monitoring
  Serial.begin(115200);

  // Timer interrupt initialization
  Timer1.initialize(round(1000000.0/read_freq));
  Timer1.attachInterrupt(timerIsr);
}

void loop() {
  // Serial.print("Frequency: ");
  // Serial.println((npeak * read_freq)/50.0);
  // Serial.println(npeak);
}

// TIMER INTERRUPT =========================================================
void timerIsr() {
  if (calibrate_step < calibrate_size)
  {
    // Store the data for threshold calibration
    calibrate[calibrate_step] = analogRead(PHOTOTRANS_PIN);
    calibrate_step++;
  }
  else
  {
    // One-time calculating the threshold using stored data from calibration phase
    if (calibrate_flag)
    {
      threshold = avg(calibrate, calibrate_size);
      Serial.print("Threshold: ");
      Serial.println(threshold);
      calibrate_flag = false;
    }

    // Read data from the light sensor
    current_val = analogRead(PHOTOTRANS_PIN);

    // Pop if the timewindow is full
    if (timewindow.isfull())
    {
      npeak = npeak - timewindow.pop();
    }
    
    // Hit-crossing storing
    if (last_val < threshold && current_val >= threshold)
    {
      timewindow.push(1);
      npeak++;
    }
    else
    {
      timewindow.push(0);
    }

    output_count++;
    if (output_count >= output_timestamp)
    {
      Serial.print("Frequency: ");
      Serial.println((npeak * read_freq)/500.0);
      output_count = 0;
    }

    last_val = current_val;
  }
}

// AUXILIARY FUNCTIONS =====================================================
// Calculate avg of an int array
float avg(int a[], int a_size)
{
  float sum = 0.0;
  for (int i = 0; i < a_size; i++)
  {
    sum += a[i];  
  }
  return sum/a_size;
}