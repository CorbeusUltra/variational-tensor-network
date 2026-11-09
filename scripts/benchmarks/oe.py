from time import perf_counter

import opt_einsum as oe

D, d, chi = 2, 4, 4
D2 = D**2
memory_limit = 2**20  # Maximum intermediate size in elements

equations = {
    "d": "ai,iuI,IA,bla,rulqsS,ArB,cLb,RqLDtT,BRC,jc,JDj,CJ->stST",
    "r": "ia,aub,bUc,cj,Ili,quldsS,RUqDtT,jRJ,AI,BdA,CDB,JC->stST",
}
shape_by_rank = {2: (chi, chi), 3: (chi, D2, chi), 6: (D2,) * 4 + (d, d)}


for direction, equation in equations.items():
    terms = equation.split("->")[0].split(",")
    shapes = tuple(shape_by_rank[len(term)] for term in terms)
    for strategy in ("greedy", "auto", "auto-hq"):
        start = perf_counter()
        path, info = oe.contract_path(equation, *shapes, shapes=True, optimize=strategy, memory_limit=memory_limit)
        search_ms = (perf_counter() - start) * 1000
        largest_mib = float(info.largest_intermediate) * 8 / 2**20
        print(f"{direction} | {strategy:7s} | search={search_ms:8.2f} ms | FLOPs={info.opt_cost:.3e} | largest={largest_mib:.3f} MiB")


# ApB = 2 * D2**4 * d**2 * 8 # bytes
# AB = D2**6 * d**4 * 8      # bytes

# ApB /= 1024**2 # MiB
# AB /= 1024**2  # MiB

# print(f"memory(A+B) = {ApB} MiB")
# print(f"memory(AB) = {AB} MiB")
