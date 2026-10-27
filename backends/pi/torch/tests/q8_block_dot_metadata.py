"""Host packing/index/integer controls only, not GLSL or device evidence."""
from pathlib import Path
import json
import re
import struct

VK=Path(__file__).resolve().parents[2]/"third_party/ggml/src/ggml-vulkan"


def read_word(data,byte):
    words=struct.unpack("<"+"I"*(len(data)//4),data)
    index=byte//4
    value=words[index]
    indices=[index]
    if byte&2:
        indices.append(index+1)
        value=((value>>16)|(words[index+1]<<16))&0xffffffff
    return value,indices


def signed(word):
    return struct.unpack("<4b",struct.pack("<I",word))


def main():
    shader=(VK/"vulkan-shaders/managed_q8_0_block_dot_x4.comp").read_text()
    header=(VK/"managed-q8-block-dot.h").read_text()
    old=(VK/"ggml-vulkan-push-constants.h").read_text()
    fields=lambda x,t:[v.strip() for names in re.findall(t+r"\s+([^;]+);",x) for v in names.split(",")]
    a=fields(old.split("struct vk_mat_vec_push_constants {")[1].split("};")[0],"uint32_t")
    b=fields(header.split("struct Parameters {")[1].split("};")[0],"uint32_t")
    c=fields(shader.split("uniform Parameters {")[1].split("} p;")[0],"uint")
    assert a==b==c and len(a)==13
    assert "layout(local_size_x=64," in shader
    assert "const uint block=gl_SubgroupInvocationID;" in shader
    visits=[4*group+sg for group in range(12967) for sg in range(4)]
    assert len(set(visits))==51868 and visits[:51865]==list(range(51865))
    assert not re.search(r"(?m)^shared\s",shader) and "barrier(" not in shader
    controls=0
    for k,m in ((32,2),(64,1),(96,2),(128,3),(384,33),(512,5)):
        blocks=k//32; count=m*blocks
        assert count%2==0
        data=bytearray()
        qs=[]
        for ib in range(count):
            q=[((ib*19+i*29)%256)-128 for i in range(32)]
            qs.append(q)
            data+=struct.pack("<e",(-1 if ib%2 else 1)*(0.125+(ib%9)/16))
            data+=struct.pack("<32b",*q)
        assert len(data)==count*34 and len(data)%4==0
        for ib in range(count):
            offset=34*ib
            scale_word=struct.unpack_from("<I",data,(offset//4)*4)[0]
            half=(scale_word>>((offset&2)*8))&65535
            assert struct.pack("<H",half)==data[offset:offset+2]
            for i in range(8):
                word,indices=read_word(data,offset+2+4*i)
                assert list(signed(word))==qs[ib][4*i:4*i+4]
                assert max(indices)*4+3<len(data)
                controls+=1
        # Canonical x4 activation ds[4] + qs[32], including partial last x4.
        activation=bytearray(((blocks+3)//4)*144)
        bqs=[]
        for block in range(blocks):
            q=[((block*13+i*17)%255)-127 for i in range(32)];bqs.append(q)
            outer,inner=divmod(block,4)
            struct.pack_into("<ee",activation,outer*144+inner*4,0.25,0.5)
            struct.pack_into("<32b",activation,outer*144+16+inner*32,*q)
        for row in range(m):
            actual=0;expected=0
            for block in range(blocks):
                subtotal=0;outer,inner=divmod(block,4)
                for i in range(8):
                    aw,_=read_word(data,(row*blocks+block)*34+2+i*4)
                    bw=struct.unpack_from("<I",activation,(outer*36+4+inner*8+i)*4)[0]
                    subtotal+=sum(a*b for a,b in zip(signed(aw),signed(bw)))
                assert abs(subtotal)<=32*128*128
                actual+=subtotal
                expected+=sum(a*b for a,b in zip(qs[row*blocks+block],bqs[block]))
            assert actual==expected
            # Flattened x/z native row split, including output tail.
            for gx in (1,7,65535):
                z,x=divmod(row,gx)
                assert x+gx*z==row
    print(json.dumps({"status":"PASS","PC_fields":13,"packed_word_controls":controls,
                      "physical_cases":6,"scope":"host packing/index/integer only; no target numerical or performance claim"}))


if __name__=="__main__":main()
