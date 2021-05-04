import os
import random

import torch

from FastSpeech2.FastSpeech2 import FastSpeech2
from FastSpeech2.FastSpeechDataset import FastSpeechDataset
from FastSpeech2.fastspeech2_train_loop import train_loop
from TransformerTTS.TransformerTTS import Transformer
from Utility.path_to_transcript_dicts import build_path_to_transcript_dict_thorsten


def run(gpu_id, resume_checkpoint, finetune):
    if gpu_id == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        device = torch.device("cpu")

    else:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_id)
        device = torch.device("cuda")

    torch.manual_seed(13)
    random.seed(13)

    print("Preparing")
    cache_dir = os.path.join("Corpora", "Thorsten")
    save_dir = os.path.join("Models", "FastSpeech2_Thorsten")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    path_to_transcript_dict = build_path_to_transcript_dict_thorsten()

    failed_alignments = ["00ef34f25b24d2ebbf90eebc4ae13e22",  # unfortunately, this had to be done manually
                         "0a5e99895d7bf2fc3d13739722bc17d6",  # by looking at the visualization of the
                         "0a75c5266fff2e761a2f77135bc4cb09",  # alignment.
                         "0ac145ba2e9a454eef7fe0ac1c48ee77", "0b0c93a0547b6b11037cd469a357e9c0", "0b297a00558496869b96e7e9be5d1211",
                         "0c433439c2c76d2e124c7944e72f080c", "0d2bdea95e732f8ceefbad7157ca48dd", "1b6df5430b05420ca5d6f14fb0c3887",
                         "1bb6bbe4bfa6e2b9e6503097b6de9394", "1c8529bbd9d1a4c9d65e9e2a7a9c0259", "1cfcab477d6fd82b277c8383fc00435b",
                         "1d582c9d4533437ec5edb726049cf338", "1efcfae7cb029fb761d2387815319c97", "1f15ffe209193800004915da1a791ff5",
                         "02c60731b27fbd0647061a4cccd89dce", "2b06f436072e97daed2579455812da6", "2bd16cf9e99408a603ae2775bf2fa18c",
                         "2beb1bcd28bc5fd9c0f45b7b36c694b2", "2db4db9a8063612e577cf23f5b468d6", "2ddfdb2b30b0ab7598b336bc54c3c877",
                         "2dfe0dbe07ac27826aa7492532957f07", "2e5cb92425609c56cd0c396d2c1521d3", "3a53b4c05a9a3db833688ab7a3329d89",
                         "3ab2c5af3e5a47ac8c18dad79e708689", "3b55950f1b5e299b3d1df8746a46489d", "3d0f71ffd5f8d0b9807ad1f8e6c9239e",
                         "3d17514e3da75d06bf5ccaed068ba7d", "3d92281a62243cbad15bd97d560f1447", "4a7673c9e1d321591441fb2c7b72a478",
                         "04e6bba280719e36f97976c033a97d90", "4a242fb7be2c49b2016c8122595b3461", "4c1c8fcd01233233c1ca0428422f935f",
                         "4c6a4c7c013c92d4a39dceffafb908c8", "4d5edbe78517eec67d7b1f5500d21739", "5bc99b906f2605f531c361a05cf2a841",
                         "5bcbf92d3e9efe628cd5c054f94fcf6", "5c7a2151af4f8e03265b40beb8f79a31", "5cc46c7cc8ec6019259a723a256bb683",
                         "5d9a4df6f63c8806d54efa10e1f42ce3", "5dfc244f5b67e3b5ea71734ecde187bd", "5ed46ca468ce13370003c2e1a5937bb3",
                         "6a7ae506b7e5689a662f84edbc4ef356", "6d800bfb287cddb0e68eb00051051965", "6ec744a85a144d1fde37684ed10518a9",
                         "6fbe3b1a68e122ed2a4a15600350d1f4", "6ff2506cd2645e491e2f889b5fc587e3", "07f28ffe176a5a4099a924acfadc6350",
                         "07f53abdf4f36f531b8408d30f7bc400", "7a89fa9677e3a523eafa2627a5574a7", "7ab223380c4fe6b20c5b668416871848",
                         "8ae211df4fd05165943c9f865d627f64", "8ae771b1d89146d7cd766cb1846d3ed7", "8b523ce3110751c812213716b63d2b2d",
                         "8bad409995bcddf47516136cde2bbcd6", "8c481b1c2ffc4e28fb1d105f32a722cd", "8d98eb444f9dbe5d7fa0aa3d39c61459",
                         "8e343b97cff378fd1b361a2878b1f0c4", "8ee034419d4ed85aed07feb167261f6", "8ef07c32041e05d6b57788502d507660",
                         "8f5690b5a231cebfcec7a8b483fc2810", "9e840ab9a0e0794d63a27460a9d4e9c4", "9f3dbb4c276d5b74c6c2188af4960d38",
                         "12dc6efe9a2ade0a285e24510bba499d", "14adb41a2e47298db190f6da7c33b3f", "21cb4fe66a45b01ecb662e905e573eb5",
                         "22b76d929f074befbb5783dae54da48b", "22fc60d17efac8b5eef466fb4c98be64", "34e4bd0eb09541d7874c3599cc0cc6dd",
                         "38da515f3effda0b748a443d67035ccb", "38db51a449df43070d8969519a5162e3", "42fe122d13fbd5865c01d60a19488875",
                         "43b0686c0704e7083d8688ed47edf74b", "44b4f11f18b9899b75141262bd65a1a6", "44bae94bf48da7881213b2c96c83b306",
                         "45cef4e64dff7a7648f8b72e6da23b0f", "48beff3903a30cc74790d3bbb82edf88", "049eb8e72543f646d772ad90f17ba09f",
                         "50d067487070b39f64f4395a730cdbcb", "54f1b04d9cbef846e9f55d7d4b7a49a9", "55c2e6a4b506c46632712bb7c95ae7be",
                         "61d602cdef5e55a7bbf5ed0d281f8558", "66fd1e244b230347f14d3410529d431c", "68f29ae45858391f00f10d5891dec46c",
                         "73bb3035916192a12492f8ec01e7380b", "0074b9d13f21f24b4b43f904fa493972", "75ce1af31561fe821011b8bf6d22fde7",
                         "76c516cf5e0a3101271421d26ab52755", "83e71d83b2dcea23dcab3b64182494cc", "84e7f21bede8b958348c7d1b967af7c8",
                         "93fb36bb9edcde5fe5e0022358f2c44b", "094e886dfd7c33aa9a3aedf46e342817", "95fefc70f2962390dba54fe00be1aecc",
                         "97d7df19998efa68f79276186395ac83", "99fcd29f82469aeae94e814e1039a16f", "130a01c148f028db4000c73bf730a142",
                         "171a3373df8e88252ff52d65a1266726", "187e94efd4c93b08b3fb6f1c64a36f0e", "201eb90026faec3acf0e4263338205c4",
                         "256c6f450dfdf1e0860a59506b41c3de", "260d8276cf3df72d52e76c9d353dcd52", "420d90df25e1d2ee67f44c0d877a2dcd",
                         "426bcddd3b3cc66fd0a27fd55945c4fb", "459f109c6106de60234d476eac5bbbc6", "488c51af15f4257bfdfa447ac1a963af",
                         "490a4786175bdc41cc35349b06bb0640", "558d61dc44db93b667435d9e873cc012", "563df8d64fe284593bbc253b30d6caac",
                         "896f37881cb22a953c2b33cc4bcc2536", "942b46d717d9da24c03476f18efd233d", "950f2b936729e6ff7df52362456b6046",
                         "969f8da6878ce490f3f50e8129ad17c3", "974bc732482e282342e73b21ca122d98", "978cec8d468c031072c973d22362111f",
                         "0981d7e905ed50346a29520af9fcc46", "1283b928f0e51184d8b8292a2e1efbe0", "1842de86353e9e8986bdce0663beca61",
                         "2006ffa2312c8845159c06a9b11a6844", "2656daa3840e2c6f4d9169f50b5891e6", "6079da393c4782931ebb1aa8ccf0285",
                         "5996d47279cacd80e25fd2016cfc9192", "6258a3a55a9c1f66128e37ba2a291761", "7376f061abe34e89a8b289c08ae68bb4",
                         "7568a4c36db2b19f05a590b244bfc021", "7372d66a73fdf2cf3ae4fc4fb789af78", "7934d5c1e12e380429a0f5a643a5d4d5",
                         "9572d074b323bb26d72f987566606564", "9863dd44b098a0a2b0c35418485a0582", "13600df781d531cd3b326a3ef2071657",
                         "013304fb448ff500ac6cd9a9eab62002", "55093ce90b27741c8c1adffd72074375", "67109eb3006f145c94ca81057ccfa608",
                         "85742e25c72b1473f77d5dca25232a64", "89615cd7608d3e3c6210e28ed20e8794", "91563d292d822e00f98a9a9147925e58",
                         "94013f7930342b23a193eba2c5ad8dff", "251782bfb5c2b57d8346250eda5eae39", "244989a39ebbf3c70545a035780ae17d",
                         "272048bd16aa403b325e0eb51368c5d1", "0812229e6dff0464c5efd5e47eef44ee", "903939e6ee8ba173e97aa2d20e53d28",
                         "001364406f288f03136403c611fff1dc", "312579479af19e1dfe8195291706ffb9", "2147857006d825d1e0ca51acb9c0fd5",
                         "3973191525bc5faf79717b08f5ce5fcf", "35094685799a19d8b4ce0e576bd92526", "a6a3aff614140a0aaf6a6165919ce8a9",
                         "a51af029695ebf34deb08b6cbcebc5d9", "a935fc2905928f651ff40fb61bc8196b", "ae28219c03cc7c5c8a8374713bb05076",
                         "af855cc0469b7e026afcca0fecc9e620", "b5a7a8be26bf5fdf9e5d9400badf4fd1", "b6cc8761dd74181dc4791de89525992b",
                         "b42a5ce0c35905f1eb18055fea15300d", "b83e44893a3d62f171b77a01d3425a4c", "b88055e13af091c04cbaf07267986feb",
                         "bcc6dbb25de75e130e55ab9eb61ddf94", "bd6e00def2f0eff846e1f9c631cba87", "bfcab5b102576be5c491ad2a41e63a35",
                         "c1e91183c4cea1fc5e895f104ddc7c22", "c3fe4d0d9d5df773d1f50706571c066b", "c4e11838c3352c9df1b0a88022ee273",
                         "c7bb05acc8c88483a1f48cb0662cb387", "c9ef0ca2d0765b80d82a7ae289995e15", "c48eefc97659be2d9f9d3a592f102e9b",
                         "c2197ce0ba9e72fccf6130882b586db2", "c756657d5ca9cc609b862d8de4df86ab", "cb0d4a4012b0037a83e2ba88d7eda429",
                         "ce2ce8fcec388b014aed3070cd1e61c7", "cf661dc3220df2dfa881f4ca5476361e", "cf648d9a9f4f37791ba52edd773e34d9",
                         "d2abc2a4a8a41019d39ba912519ae4ac", "d3dd271a85c6af1c5a5214e080b37ddc", "d9f4c3cffa7b905452dcb4b7f2e1326f",
                         "d9e0d19e7301d807033915ae5206fffe", "d282c46783d01376e3919706affe6bd5", "d93632dfa98e6d2db697bcc25033f160",
                         "daad336c74b3e7e9234a0569cdb7936d", "dbcdb51f0580558b315f0e43b83d9950", "e0aa021e21dddbd6d8cecec71e9cf564",
                         "e121bf242048de7030aa9dbaccf63ca1", "e91f997a01fefe877e6511754ab8cf20", "e768c3fdf55ffee46ab2fc65cb8d0713",
                         "e934f30fa049a9d13682572005c7f55b", "e7093e38883b3ade9e9b6576b89ebcc4", "e7933316da8062c247c337420e3b36e1",
                         "ea7183bc5ff137d16632b9a166231a3c", "eed545154cd0ba59a778ba8fedf63801", "ef7e9a64f2ae79c989ad125163c110fc",
                         "ef4341145157a5a8232049fcf1872a6d", "f54e58899b69f92725b2e4f4e4fcd5c0", "f250d2cdd0acb4fef7fd42eb5ed1a6d5",
                         "f03094e1bb5d3c6e900d64f983cdb7ef", "fd5afeb772fa18652b79594adaf3c3c8", "fd07c9b4b793f7d3f0927fc32d40db4b",
                         "fd539ca17a17132049c4fef4619df638", "fe1355ced47acae49afaeb6f2ce55897", "fedd2a78e49fad7459e930a78287731f",
                         "ff6d0dc84580d79ee31e8d08c5ddaf59", "ff15289ef1363ee446020c6e1f1368b4", "ffdba2cc82f42df7dfd3fa6cf44d3587",
                         "fffa652d39eebcbaccce1e67ee5d5bcb"]

    path_blacklist = ["/mount/resources/speech/corpora/Thorsten_DE/wavs/" + x + ".wav" for x in failed_alignments]

    acoustic_model = Transformer(idim=133, odim=80, spk_embed_dim=None)
    acoustic_model.load_state_dict(torch.load(os.path.join("Models", "TransformerTTS_Thorsten", "best.pt"),
                                              map_location='cpu')["model"])

    train_set = FastSpeechDataset(path_to_transcript_dict,
                                  cache_dir=cache_dir,
                                  acoustic_model=acoustic_model,
                                  diagonal_attention_head_id=12,
                                  lang="de",
                                  min_len_in_seconds=1,
                                  max_len_in_seconds=10,
                                  device=device,
                                  path_blacklist=path_blacklist)

    model = FastSpeech2(idim=133, odim=80, spk_embed_dim=None)

    print("Training model")
    train_loop(net=model,
               train_dataset=train_set,
               device=device,
               save_directory=save_dir,
               steps=400000,
               batch_size=32,
               gradient_accumulation=1,
               epochs_per_save=10,
               use_speaker_embedding=False,
               lang="de",
               lr=0.001,
               warmup_steps=8000,
               path_to_checkpoint=resume_checkpoint,
               fine_tune=finetune)
