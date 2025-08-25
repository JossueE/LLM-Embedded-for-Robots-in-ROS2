#ifndef WHISPER_UTIL__AUDIO_BUFFERS_HPP_
#define WHISPER_UTIL__AUDIO_BUFFERS_HPP_

#include <chrono>
#include <limits>
#include <mutex>
#include <vector>

#include "whisper.h"

namespace whisper {
inline std::size_t time_to_count(const std::chrono::milliseconds &ms) {
  return ms.count() * WHISPER_SAMPLE_RATE / 1e3;
};

inline std::chrono::milliseconds count_to_time(const std::size_t &count) {
  return std::chrono::milliseconds(count * static_cast<std::size_t>(1e3) / WHISPER_SAMPLE_RATE);
};

inline std::chrono::nanoseconds count_to_time_ns(const std::size_t &count) {
  return std::chrono::nanoseconds(count * static_cast<size_t>(1e9) / WHISPER_SAMPLE_RATE);
};

/**
 * @brief A ring buffer implementation. This buffer is **not** thread-safe. It is the user's
 * responsibility to ensure thread-safety.
 *
 * @tparam value_type
 */
template <typename value_type> class RingBuffer {
protected:
  using const_reference = const value_type &;

public:
  RingBuffer(const std::size_t &capacity);

  void enqueue(const_reference data);
  value_type dequeue();
  inline bool is_full() const { return size_ == capacity_; }
  inline bool almost_full() const { return size_ + 1 == capacity_; }
  inline bool empty() const { return size_ == 0; }
  void clear();

  inline const std::size_t &capacity() const { return capacity_; }
  inline const std::size_t &size() const { return size_; }

protected:
  void increment_head_();
  virtual void increment_tail_();

  std::size_t capacity_;
  std::vector<value_type> buffer_;
  std::size_t head_;
  std::size_t tail_;
  std::size_t size_;
};

/**
 * @brief Implementation of thread-safe behavior.
 * Inherits from RingBuffer and adds mutex protection for concurrent access.
 *
 * @tparam value_type
 */
template <typename value_type>
class ThreadSafeRing : public RingBuffer<value_type> {
public:
  ThreadSafeRing(const std::size_t& capacity);

  void enqueue(const typename RingBuffer<value_type>::const_reference data);
  void enqueue(const std::vector<value_type>& data); 

  value_type dequeue();

  void clear();

protected:
  // mutable keyword: Allow mutex to be grabbed in const context
  mutable std::mutex mutex_;
};

/**
 * @brief A thread-safe buffer for storing audio data. The user enqueues data from an audio stream
 * in thread A and reads a copy of the data (peak) in thread B. 
 * When buffer is full overwrite oldest data, so buffer contents are always the newest data in
 * the stream.
 *
 * Thread A: enqueue into _buffer 
 * Thread B: peak (copy) in-order from _buffer 
 *
 */
class AudioRing : public ThreadSafeRing<std::int16_t> {
private:
  // Keep a timestamp of the start of the buffer.
  std::chrono::system_clock::time_point audio_start_;
  const std::chrono::nanoseconds time_inc_;
  bool audio_start_set;

protected:
  // Overwrite this function to also increase the buffer audio_start_ timestamp when dropping data
  void increment_tail_() override;

public:
  AudioRing(const std::chrono::milliseconds &buffer_capacity,
                          std::chrono::system_clock::time_point cur_time);
  AudioRing(const std::chrono::milliseconds &buffer_capacity);

  void set_start_timestamp(std::chrono::system_clock::time_point cur_time);
  std::chrono::system_clock::time_point get_start_timestamp() const;

  // Get a logical order copy of all data in ring buffer, along with the timestamp
  std::tuple<std::vector<float>, std::chrono::system_clock::time_point> peak() const;

  // Add zeros so the audio_start_ + count_to_time(size_) = cur_time.
  //     :return: The number of zeros added.
  size_t decay(std::chrono::system_clock::time_point cur_time);

  void clear();

  bool is_audio_start_set() const {
    return audio_start_set;
  }
};

/**
 * Implementations -- RingBuffer.cpp
**/

template <typename value_type>
RingBuffer<value_type>::RingBuffer(const std::size_t &capacity)
    : capacity_(capacity), buffer_(capacity) {
  clear();
};

template <typename value_type> void RingBuffer<value_type>::enqueue(const_reference data) {
  increment_head_();
  if ( is_full() ) {
    increment_tail_();
  }
  buffer_[head_] = data;
}

template <typename value_type> value_type RingBuffer<value_type>::dequeue() {
  increment_tail_();
  return buffer_[tail_];
}

template <typename value_type> void RingBuffer<value_type>::clear() {
  head_ = 0;
  tail_ = 0;
  size_ = 0;
}

template <typename value_type> void RingBuffer<value_type>::increment_head_() {
  ++head_;
  ++size_;
  if ( head_ >= capacity_ ) {
    head_ = 0;
  }
}
template <typename value_type> void RingBuffer<value_type>::increment_tail_() {
  ++tail_;
  --size_;
  if ( tail_ >= capacity_ ) {
    tail_ = 0;
  }
}

/**
 * Implementations -- ThreadSafeRing.cpp
**/

template <typename value_type>
ThreadSafeRing<value_type>::ThreadSafeRing(const std::size_t& capacity)
    : RingBuffer<value_type>(capacity) {
}

template <typename value_type>
void ThreadSafeRing<value_type>::enqueue(const typename RingBuffer<value_type>::const_reference data) {
  std::lock_guard<std::mutex> lock(mutex_);
  RingBuffer<value_type>::enqueue(data);
}

template <typename value_type>
void ThreadSafeRing<value_type>::enqueue(const std::vector<value_type>& data) {
  std::lock_guard<std::mutex> lock(mutex_);  // Lock the mutex once for the entire operation
  for (const auto& sample : data) {
    RingBuffer<value_type>::enqueue(sample);  // Enqueue each element
  }
}

template <typename value_type>
value_type ThreadSafeRing<value_type>::dequeue() {
  std::lock_guard<std::mutex> lock(mutex_);
  return RingBuffer<value_type>::dequeue();
}

template <typename value_type>
void ThreadSafeRing<value_type>::clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  RingBuffer<value_type>::clear();
}

} // end of namespace whisper
#endif // WHISPER_UTIL__AUDIO_BUFFERS_HPP_

