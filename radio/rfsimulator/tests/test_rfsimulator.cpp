#include <gtest/gtest.h>
extern "C" {
#include "rfsimulator.h"
void exit_function(const char *file, const char *function, const int line, const char *s, const int assert)
{
  if (assert) {
    abort();
  }
  exit(EXIT_SUCCESS);
}
configmodule_interface_t *uniqCfg;
void *get_shlibmodule_fptr(char *modname, char *fname)
{
  return nullptr;
}
#include "openair1/SIMULATION/TOOLS/sim.h"
#include "nr_common.h"
extern uint32_t signal_energy_nodc(const c16_t *input, uint32_t length);
}
#include <algorithm>
#include <cmath>

channel_desc_t *get_awgn_channel_106_rb(void)
{
  double sample_rate;
  unsigned int samples_per_frame;
  double tx_bw;
  double rx_bw;
  double DS_TDL = .03;
  get_samplerate_and_bw(1, 106, 0, &sample_rate, &samples_per_frame, &tx_bw, &rx_bw);
  auto channel = new_channel_desc_scm(1, 1, AWGN, sample_rate / 1e6, 0, tx_bw, DS_TDL, 0.0, CORR_LEVEL_LOW, 0, 0, 0, -40);
  random_channel(channel, 0);
  return channel;
}

const int FS = INT16_MAX;
const float rms_fs = 0.707 * FS * FS;

float calculate_power(std::vector<c16_t> in, int reference_power)
{
  int32_t rms = signal_energy_nodc(in.data(), in.size());
  return reference_power + 10 * log10(rms / rms_fs);
}

TEST(rfsimulator, add_rx_input)
{
  const int16_t sample_values = 256;
  int size = 12 * 50;
  std::vector<c16_t> input;
  input.resize(size);
  std::fill(input.begin(), input.end(), (c16_t){sample_values, sample_values});
  std::vector<c16_t> output;
  output.resize(size);
  std::fill(output.begin(), output.end(), (c16_t){0, 0});
  int rxAnt = 0;

  channel_desc_t *channel = get_awgn_channel_106_rb();
  int nbSamples = input.size();
  uint64_t TS = 0;
  uint32_t CirSize = sizeof(input);
  int16_t tx_power_reference = 0;
  int16_t rx_power_reference = 0;
  rxAddInput(input.data(), output.data(), rxAnt, channel, nbSamples, TS, CirSize, tx_power_reference, rx_power_reference);

  std::cout << "Input : " << calculate_power(input, tx_power_reference) << " [dBm] " << std::endl;
  std::cout << "Output (unchanged): " << calculate_power(output, rx_power_reference) << " [dBm]" << std::endl;

  channel->path_loss_dB = -3;
  random_channel(channel, 0);
  std::fill(output.begin(), output.end(), (c16_t){0, 0});

  rxAddInput(input.data(), output.data(), rxAnt, channel, nbSamples, TS, CirSize, tx_power_reference, rx_power_reference);
  std::cout << "Output (3dB loss): " << calculate_power(output, rx_power_reference) << " [dBm]" << std::endl;

  channel->path_loss_dB = 3;
  random_channel(channel, 0);
  std::fill(output.begin(), output.end(), (c16_t){0, 0});

  rxAddInput(input.data(), output.data(), rxAnt, channel, nbSamples, TS, CirSize, tx_power_reference, rx_power_reference);
  std::cout << "Output (3dB gain): " << calculate_power(output, rx_power_reference) << " [dBm]" << std::endl;

  channel->path_loss_dB = 3;
  std::fill(output.begin(), output.end(), (c16_t){0, 0});

  rxAddInput(input.data(), output.data(), rxAnt, channel, nbSamples, TS, CirSize, tx_power_reference, rx_power_reference);
  std::cout << "Output (3dB gain) [no channel regeneration]: " << calculate_power(output, rx_power_reference) << " [dBm]"
            << std::endl;

  channel->path_loss_dB = 0;
  std::fill(output.begin(), output.end(), (c16_t){0, 0});

  tx_power_reference -= 3;
  rxAddInput(input.data(), output.data(), rxAnt, channel, nbSamples, TS, CirSize, tx_power_reference, rx_power_reference);
  std::cout << "Output (3dB gain) [increased tx power]: " << calculate_power(output, rx_power_reference) << " [dBm]" << std::endl;

  channel->path_loss_dB = 0;
  std::fill(output.begin(), output.end(), (c16_t){0, 0});
  tx_power_reference = 0;
  rx_power_reference = -3;
  rxAddInput(input.data(), output.data(), rxAnt, channel, nbSamples, TS, CirSize, tx_power_reference, rx_power_reference);
  std::cout << "Output (0dB gain) [increased rx sensitivity]: " << calculate_power(output, rx_power_reference) << " [dBm]"
            << std::endl;

  // Testing TX power calculation
  int16_t epre_ssb = -106;
  tx_power_reference = epre_ssb + 10 * log10((float)(rms_fs) / (sample_values * sample_values * 2));
  std::cout << "Input tx_power (at epre == -106dBm): " << calculate_power(input, tx_power_reference) << " [dBm]" << std::endl;
  std::fill(output.begin(), output.end(), (c16_t){0, 0});

  rx_power_reference = 0;
  channel->path_loss_dB = 0;
  channel->noise_power_dB = -20;
  std::fill(output.begin(), output.end(), (c16_t){0, 0});
  std::fill(input.begin(), input.end(), (c16_t){0, 0});
  rxAddInput(input.data(), output.data(), rxAnt, channel, nbSamples, TS, CirSize, tx_power_reference, rx_power_reference);
  std::cout << "Output (0dB gain, " << channel->noise_power_dB << "dBm noise): " << calculate_power(output, rx_power_reference)
            << " [dBm]" << std::endl;

  std::fill(input.begin(), input.end(), (c16_t){0, 0});
  std::fill(output.begin(), output.end(), (c16_t){0, 0});
  rxAddInput(input.data(), output.data(), rxAnt, channel, nbSamples, TS, CirSize, 3, rx_power_reference);
  std::cout << "Output (3dB gain, " << channel->noise_power_dB << "dBm noise): " << calculate_power(output, rx_power_reference)
            << " [dBm]" << std::endl;

  channel->path_loss_dB = 0;
  channel->noise_power_dB = -30;
  random_channel(channel, 0);
  tx_power_reference = 0;
  auto linear_gain = std::pow(10, 3 / 20.0);
  int16_t sample = sample_values * linear_gain;
  std::fill(input.begin(), input.end(), (c16_t){sample, sample});
  std::fill(output.begin(), output.end(), (c16_t){0, 0});

  rxAddInput(input.data(), output.data(), rxAnt, channel, nbSamples, TS, CirSize, tx_power_reference, rx_power_reference);
  std::cout << "Output (3dB) [digital sample gain]: " << calculate_power(output, rx_power_reference) << " [dBm]" << std::endl;

  free_channel_desc_scm(channel);
}

int main(int argc, char **argv)
{
  logInit();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
