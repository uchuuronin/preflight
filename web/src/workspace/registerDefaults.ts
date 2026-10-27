/** C's own entries in the registry: the four readouts and the views C ships. */
import { registerPanel, registerReadout } from "../registry";
import { MemoryStackView } from "./views/MemoryStack";

registerReadout({ id: "memory", targetId: "peak_memory", titleKey: "readout.memory", order: 10 });
registerReadout({ id: "time", targetId: "time_to_finish", titleKey: "readout.time", order: 20 });
registerReadout({ id: "cost", targetId: "total_cost", titleKey: "readout.cost", order: 30 });
registerReadout({ id: "risk", targetId: "divergence_risk", titleKey: "readout.risk", order: 40 });

registerPanel({
	id: "memory-stack",
	slot: "views",
	titleKey: "view.memory.title",
	lenses: ["finetune", "pretrain", "serving"],
	order: 20,
	component: MemoryStackView,
});
